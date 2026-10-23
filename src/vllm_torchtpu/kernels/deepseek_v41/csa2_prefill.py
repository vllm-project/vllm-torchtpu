# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ragged prefill with query-tile grids and compiler-managed DMA buffering."""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _tile_plan(starts: jax.Array, total_tokens: int, bq: int):
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


def _window(tile, tiles, positions, pages, bounds, *, page_size, non_causal):
    request, start, count = tiles[tile, 0], tiles[tile, 1], tiles[tile, 2]
    active = request < pages.shape[0]
    request = jnp.minimum(request, pages.shape[0] - 1)
    position = positions[start]
    left = bounds[request, 0] if non_causal else jnp.maximum(0, position - 127)
    right = bounds[request, 1] if non_causal else position + count
    anchor = left // page_size * page_size
    left = jnp.maximum(left, pages[request, 1])
    return position, left, jnp.where(active, right, left), anchor


def _prefill_kernel(
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
    swa_kv_ref: Any,  # VMEM: tuple of [1, page_size, 512] BF16 page slices
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
    cache_page_size: int,
    num_banks: int,
    low_pages: int,
    scale: float,
    non_causal: bool,
) -> None:
    num_pages = 256 // page_size
    tile = pl.program_id(0)
    request, t0 = tiles_ref[tile, 0], tiles_ref[tile, 1]
    position_start, left, right, anchor = _window(
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
    valid_count = tiles_ref[tile, 2]
    position_start -= row_offset
    t0 = read_start
    active = request < swa_page_ref.shape[0]
    request = jnp.minimum(request, swa_page_ref.shape[0] - 1)
    key_positions = anchor + jnp.arange(256)
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
        mask = jnp.where(low, 0xFFFF, 0) | jnp.where(high, -65536, 0)
        words = pltpu.bitcast(values, jnp.int32) & mask[:, None]
        swa_ref[rows] = pltpu.bitcast(words, jnp.bfloat16)
    num_heads = q_ref.shape[1]
    mxu_heads = pl.cdiv(num_heads, 16) * 16
    mrb_stride = mxu_heads // 4
    # Separate QK/PV ranges let the next QK overlap the preceding PV.
    # Each range holds block_q results; together they must fit 256 MRB rows.

    swa_capacity = 256 // mrb_stride
    accumulator_slots = max(2 * block_q, min(bq, swa_capacity))

    accumulators = tuple(
        jax.empty_ref(
            jax.ShapeDtypeStruct((mxu_heads, 256), jnp.float32),
            memory_space=pltpu.ACC(mxu),
        )
        for mxu in range(2)
        for _ in range(accumulator_slots)
    )

    def accumulator(mxu, address):
        return accumulators[mxu * accumulator_slots + address // mrb_stride]

    groups = [
        tuple(range(first, min(first + block_q, bq))) for first in range(0, bq, block_q)
    ]

    def swa_launch(first, size):
        for mxu in range(2):
            ks = pl.ds(mxu * 256, 256)
            pltpu.matmul_push_rhs(swa_ref[:, ks], 0, mxu, transpose=True)
            for r in range(size):
                lhs = jnp.pad(
                    q_ref[first + r, :, ks], ((0, mxu_heads - num_heads), (0, 0))
                )
                pltpu.matmul_acc_lhs(
                    accumulator(mxu, r * mrb_stride),
                    lhs,
                    load_staged_rhs=0 if r == 0 else None,
                )

    def swa_pop(first, size):
        for r in range(size):
            halves = [
                pltpu.matmul_pop(accumulator(mxu, r * mrb_stride))[:num_heads]
                for mxu in range(2)
            ]
            position = position_start + first + r
            start = left if non_causal else jnp.maximum(left, position - 127)
            end = right if non_causal else jnp.minimum(right, position + 1)
            valid = (key_positions >= start) & (key_positions < end)
            swa_score_ref[first + r] = (halves[0] + halves[1]) * (
                scale * 1.4426950408889634
            ) + jnp.where(valid[None, :], 0.0, -jnp.inf)

    def main_unpack(indices):
        # Larger tiles let the compiler rematerialize FP8 conversion at both
        # consumers. Sixteen rows retain one decode in the measured native LLO.
        for i in indices:
            for row in range(0, 512, 16):
                rows = pl.ds(row, 16)
                words = main_cache_ref[i, rows, :]
                for group in range(4):
                    shifted = lax.shift_right_logical(words, 8 * group)
                    bits = shifted.astype(jnp.uint8)
                    fp8 = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
                    values = fp8.astype(jnp.bfloat16)
                    main_kv_ref[i, rows, pl.ds(group * 128, 128)] = values

    def main_qk(indices):
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
                    accumulator(mxu, r * mrb_stride),
                    lhs,
                    load_staged_rhs=tile_index % 2,
                )

    def pop_qk(indices):
        logits = []
        for r, i in enumerate(indices):
            main_logits = jnp.concatenate(
                [
                    pltpu.matmul_pop(accumulator(mxu, r * mrb_stride))[:num_heads]
                    for mxu in range(2)
                ],
                axis=1,
            )
            logits.append(main_logits)
        return logits

    def softmax(indices, qk_values):
        for r, i in enumerate(indices):
            main_logits = qk_values[r]
            # Scale each key column after QK, using FP32 KV scales.
            kv_scales = main_scale_ref[i]
            main_logits = main_logits * (kv_scales * (scale * 1.4426950408889634))

            sink = sink_ref[...] * 1.4426950408889634
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
                    accumulator(mxu, (block_q + r) * mrb_stride),
                    lhs,
                    load_staged_rhs=tile_index % 2,
                )

        for mxu in range(2):
            pltpu.matmul_push_rhs(swa_ref[:, pl.ds(mxu * 256, 256)], 0, mxu)
            for r, i in enumerate(indices):
                lhs = jnp.pad(swa_p_ref[i], ((0, mxu_heads - num_heads), (0, 0)))
                pltpu.matmul_acc_lhs(
                    accumulator(mxu, (block_q + r) * mrb_stride),
                    lhs,
                    load_staged_rhs=0 if r == 0 else None,
                )

    def inverse_rope(indices):
        for r, i in enumerate(indices):
            output = jnp.concatenate(
                [
                    pltpu.matmul_pop(accumulator(mxu, (block_q + r) * mrb_stride))[
                        :num_heads
                    ]
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
            rotated = tail * rope_ref[i, 0][None, :] + partner * rope_ref[i, 1][None, :]
            valid = active & (i >= row_offset) & (i < row_offset + valid_count)
            output_ref[i, :, :384] = jnp.where(valid, output[:, :384], 0).astype(
                jnp.bfloat16
            )
            output_ref[i, :, 384:] = jnp.where(valid, rotated, 0).astype(jnp.bfloat16)

    for first in range(0, bq, swa_capacity):
        size = min(swa_capacity, bq - first)
        swa_launch(first, size)
        if first == 0:
            main_unpack(groups[0])
        swa_pop(first, size)
    main_qk(groups[0])
    for step, indices in enumerate(groups):
        logits = pop_qk(indices)
        if step + 1 < len(groups):
            main_unpack(groups[step + 1])
            main_qk(groups[step + 1])
        softmax(indices, logits)
        if step:
            inverse_rope(groups[step - 1])
        fused_pv(indices)
    inverse_rope(groups[-1])


def _reference(
    q,
    words,
    scales,
    banks,
    starts,
    positions,
    sink,
    rope,
    pages,
    bounds,
    lengths,
    scale,
):
    tokens = q.shape[0]
    request = jnp.minimum(
        jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right"),
        pages.shape[0] - 1,
    )
    page_size = banks[0].shape[1]
    left = jnp.maximum(0, positions - 127) if bounds is None else bounds[request, 0]
    right = positions + 1 if bounds is None else bounds[request, 1]
    keys = left[:, None] + jnp.arange(128 if bounds is None else 192)
    columns = (keys - pages[request, 0, None]) // page_size
    valid = (keys < right[:, None]) & (keys >= pages[request, 1, None])
    valid &= (columns >= 0) & (columns < pages.shape[1] - 2)
    columns = jnp.clip(columns, 0, pages.shape[1] - 3)
    page_ids = pages[request[:, None], columns + 2]
    swa = banks[0][jnp.clip(page_ids, 0, banks[0].shape[0] - 1), keys % page_size]
    if len(banks) == 2:
        high = banks[1][
            jnp.clip(page_ids - banks[0].shape[0], 0, banks[1].shape[0] - 1),
            keys % page_size,
        ]
        swa = jnp.where((page_ids < banks[0].shape[0])[..., None], swa, high)
    swa = jnp.where(valid[..., None], swa, 0)
    bits = (words[:, :, None, :] >> (8 * jnp.arange(4))[None, None, :, None]).astype(
        jnp.uint8
    )
    main = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(jnp.bfloat16)
    main = main.reshape(tokens, 512, 512)
    main_logits = jnp.einsum(
        "thd,tkd->thk", q, main, preferred_element_type=jnp.float32
    )
    main_logits *= scales[:, None, :] * scale
    main_logits = jnp.where(
        jnp.arange(512) < lengths[:, None, None], main_logits, -jnp.inf
    )
    swa_logits = (
        jnp.einsum("thd,tkd->thk", q, swa, preferred_element_type=jnp.float32) * scale
    )
    swa_logits = jnp.where(valid[:, None, :], swa_logits, -jnp.inf)
    logits = jnp.concatenate((main_logits, swa_logits), axis=-1)
    maximum = jnp.maximum(logits.max(axis=-1, keepdims=True), sink[None, :, None])
    weights = jnp.exp(logits - maximum)
    weights /= weights.sum(axis=-1, keepdims=True) + jnp.exp(
        sink[None, :, None] - maximum
    )
    main_p = (weights[:, :, :512] * scales[:, None, :]).astype(jnp.bfloat16)
    swa_p = weights[:, :, 512:].astype(jnp.bfloat16)
    output = jnp.einsum(
        "thk,tkd->thd", main_p, main, preferred_element_type=jnp.float32
    )
    output += jnp.einsum("thk,tkd->thd", swa_p, swa, preferred_element_type=jnp.float32)
    tail = output[:, :, 384:]
    tail = (
        tail * rope[:, None, 0, :]
        + tail[:, :, jnp.arange(128) ^ 1] * rope[:, None, 1, :]
    )
    output = jnp.concatenate((output[:, :, :384], tail), axis=-1).astype(jnp.bfloat16)
    return jnp.where((jnp.arange(tokens) < starts[-1])[:, None, None], output, 0)


@functools.partial(
    jax.jit, static_argnames=("scale", "compress_ratio", "bq", "interpret")
)
def csa2_prefill(
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
    bq: int = 16,
    interpret: bool = False,
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

    XLA pipelines fixed query windows and paged SWA through two VMEM buffers.
    Ragged tails may read extra queries; only valid output rows are compacted.
    interpret selects an independent JAX math reference, without MXU operations.
    """
    if q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("CSA2 prefill q must be [T,H,512]")
    tokens, heads, _ = q.shape
    if heads < 8 or heads > 128 or heads % 8:
        raise ValueError("CSA2 prefill requires H divisible by 8, in [8,128]")
    if type(bq) is not int or not 1 <= bq <= 16:
        raise ValueError("CSA2 prefill bq must be a static integer in [1,16]")
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
                f"CSA2 prefill expects {shape} {dtype}, got {value.shape} {value.dtype}"
            )
    if swa_bounds is not None and (
        swa_bounds.shape != (batch, 2) or swa_bounds.dtype != jnp.int32
    ):
        raise ValueError("SWA bounds must be INT32 [B,2]")
    if not tokens:
        return jnp.zeros_like(q)
    if interpret:
        return _reference(
            q,
            main_kv,
            main_kv_scale,
            banks,
            starts,
            positions,
            sink,
            rope_cache,
            swa_metadata,
            swa_bounds,
            main_lengths,
            scale,
        )
    bq = min(bq, tokens)
    direct_output = batch == 1 and tokens % bq == 0
    if direct_output:
        offsets = jnp.arange(tokens // bq, dtype=jnp.int32) * bq
        counts = jnp.clip(starts[-1] - offsets, 0, bq)
        tiles = jnp.stack(
            ((offsets >= starts[-1]).astype(jnp.int32), offsets, counts), 1
        )
        num_tiles = tokens // bq
    else:
        tiles, num_tiles = _tile_plan(starts, tokens, bq)
    non_causal = swa_bounds is not None
    bounds = swa_bounds if non_causal else jnp.zeros((batch, 2), jnp.int32)
    scalars = (tiles, positions, main_lengths, swa_metadata, bounds)
    page_size = min(cache_page_size, 64)
    num_pages = 256 // page_size

    def swa_map(
        tile,
        tiles_ref,
        positions_ref,
        lengths_ref,
        pages_ref,
        bounds_ref,
        *,
        block,
        bank,
    ):
        request = jnp.minimum(tiles_ref[tile, 0], batch - 1)
        _, left, right, anchor = _window(
            tile,
            tiles_ref,
            positions_ref,
            pages_ref,
            bounds_ref,
            page_size=page_size,
            non_causal=non_causal,
        )
        position = anchor + block * page_size
        column = (position - pages_ref[request, 0]) // cache_page_size
        page = pages_ref[request, jnp.clip(column, 0, pages_ref.shape[1] - 3) + 2]
        valid = (position < right) & (position + page_size > left)
        valid &= (column >= 0) & (column < pages_ref.shape[1] - 2)
        if len(banks) == 2:
            valid &= (
                (page < banks[0].shape[0]) if bank == 0 else (page >= banks[0].shape[0])
            )
            if bank == 1:
                page -= banks[0].shape[0]
        return (
            jnp.where(valid, page, 0),
            jnp.where(valid, position % cache_page_size // page_size, 0),
            0,
        )

    kernel = functools.partial(
        _prefill_kernel,
        bq=bq,
        block_q=min(6, bq, 128 // (pl.cdiv(heads, 16) * 4)),
        page_size=page_size,
        cache_page_size=cache_page_size,
        num_banks=len(banks),
        low_pages=banks[0].shape[0],
        scale=scale,
        non_causal=non_causal,
    )

    def query_map(tile, tiles_ref, *_):
        return jnp.minimum(tiles_ref[tile, 1], tokens - bq), 0, 0

    def query_spec(shape):
        return pl.BlockSpec(
            tuple(pl.Element(n) for n in (bq, *shape)),
            query_map,
            pipeline_mode=pl.Buffered(2),
            memory_space=pltpu.VMEM,
        )

    tiled_output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tiles.shape[0] * bq, heads, 512), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(num_tiles,),
            in_specs=[
                query_spec((heads, 512)),
                query_spec((512, 128)),
                query_spec((1, 512)),
                tuple(
                    pl.BlockSpec(
                        (1, page_size, 512),
                        functools.partial(swa_map, block=block, bank=bank),
                        pipeline_mode=pl.Buffered(2),
                        memory_space=pltpu.VMEM,
                    )
                    for block in range(num_pages)
                    for bank in range(len(banks))
                ),
                pl.BlockSpec(
                    (heads, 1),
                    lambda *_: (0, 0),
                    pipeline_mode=pl.Buffered(1),
                    memory_space=pltpu.VMEM,
                ),
                query_spec((2, 128)),
            ],
            out_specs=pl.BlockSpec(
                (bq, heads, 512),
                lambda tile, *_: (tile, 0, 0),
                pipeline_mode=pl.Buffered(2),
                memory_space=pltpu.VMEM,
            ),
            scratch_shapes=(
                pltpu.VMEM((bq, 512, 512), jnp.bfloat16),
                pltpu.VMEM((256, 512), jnp.bfloat16),
                pltpu.VMEM((bq, heads, 256), jnp.float32),
                pltpu.VMEM((bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, heads, 256), jnp.bfloat16),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="csa2_prefill",
    )(
        *scalars,
        q,
        main_kv,
        main_kv_scale[:, None, :],
        tuple(bank for _ in range(num_pages) for bank in banks),
        sink[:, None],
        rope_cache,
    )
    if direct_output:
        return tiled_output
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
