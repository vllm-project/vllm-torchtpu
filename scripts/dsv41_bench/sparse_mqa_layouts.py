# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental token-major sparse MQA scorer.

Adapted from index_selection/fp8/sparse_logits.py. The query/MXU/output
pipeline is unchanged; only packed K decoding and scale reads differ.
Kept in the benchmark directory until cache writer/reader integration is measured.
"""

from dataclasses import replace
from functools import partial, reduce
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import PAGE_RECORDS
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.common import (
    QueryTile,
    _dma_rows,
    _head_order,
    _next_query_tile,
    _reduce_heads,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.tuned import (
    PACKED_FP8_PREPARED_MQA_CONFIG,
    PACKED_FP8_PREPARED_PREFIX_TOKENS,
)


def _kernel(
    offsets,
    windows,
    query_start,
    inputs,
    out,
    scratch,
    *,
    config,
    info,
    fields,
    loop_tile,
    byte_transpose,
    native_packing,
    lane_repack,
):
    q, weights, pools = inputs
    queries, keys, scores, counts, read_sem, write_sem, q_sem = scratch
    keys = tuple(pltpu.annotate(key, no_store=True) for key in keys)
    weights = pltpu.annotate(weights, no_store=True)
    tokens, heads, dim = q.shape
    q_tile, mxu = config.query_tile, info.mxu_column_size
    begin, width = query_start[()], out.shape[1]
    group_pages = pools[0].shape[0] // fields // (offsets.shape[0] - 1)
    next_query = partial(
        _next_query_tile, offsets=offsets, tile_size=q_tile, end=begin + out.shape[0]
    )
    lanes = jnp.arange(PAGE_RECORDS)
    owners = _head_order(
        jnp.repeat(jnp.arange(q_tile), heads)[:, None], q_tile, info.num_sublanes
    )[:, 0]
    selector = (jnp.arange(q_tile)[:, None] == owners[None, :]).astype(jnp.bfloat16)

    def rows(query):
        return [
            jnp.minimum(offsets[query.page_row] + i, tokens - 1) for i in range(q_tile)
        ]

    def limit(query):
        return reduce(
            jnp.maximum,
            [
                jnp.where(
                    (query.count > 0)
                    & (i < offsets[query.page_row + 1] - offsets[query.page_row]),
                    jnp.maximum(windows[r, 2], windows[r, 3]),
                    0,
                )
                for i, r in enumerate(rows(query))
            ],
        )

    def load_query(query, step, wait=False):
        destination = queries.at[pl.ds((step & 1) * q_tile, q_tile)]
        if not wait:
            destination[...] = jnp.zeros(destination.shape, destination.dtype)
        _dma_rows(
            q,
            destination,
            q_sem.at[step & 1],
            jnp.where(
                query.count > 0,
                offsets[query.page_row + 1] - offsets[query.page_row],
                0,
            ),
            wait,
            src_start=offsets[query.page_row],
        )

    def load_keys(query, step, wait=False):
        pages = pl.cdiv(limit(query), PAGE_RECORDS)

        @pl.when(pages > 0)
        def transfer():
            for stream, rows in enumerate((fields,) if fields == 33 else (32, 1)):
                copy = pltpu.make_async_copy(
                    pools[stream].at[
                        pl.ds(query.page_row * group_pages * rows, pages * rows)
                    ],
                    keys[stream].at[step % config.key_buffers, pl.ds(0, pages * rows)],
                    read_sem.at[step % config.key_buffers, stream],
                )
                (copy.wait if wait else copy.start)()

    def output(slot):
        return scores.at[slot, :, pl.ds(0, width)]

    def release(slot):
        src = output(slot)
        _dma_rows(src, src, write_sem.at[slot], counts[slot], wait=True)

    def body(state):
        query, following, lookahead, step = state
        slot = step & 1
        load_query(query, step, wait=True)
        load_keys(query, step, wait=True)
        release(slot)
        load_query(following, step + 1)
        load_keys(lookahead, step + 2)
        scores[slot] = jnp.full(scores.shape[1:], -jnp.inf, jnp.float32)
        query_rows = rows(query)
        weight = jnp.concatenate([weights[r][:, None] for r in query_rows])
        head_weights = _head_order(weight, q_tile, info.num_sublanes)
        lhs = _head_order(
            queries[pl.ds(slot * q_tile, q_tile)].reshape(q_tile * heads, dim),
            q_tile,
            info.num_sublanes,
        )

        def key_rows(start, count):
            return keys[0][step % config.key_buffers, pl.ds(start, count)]

        bounds = [
            jnp.stack([windows[row, i] for row in query_rows])[:, None]
            for i in range(4)
        ]

        def store_page(page, values):
            positions = page * PAGE_RECORDS + lanes[None, :]
            start, split, end1, end2 = bounds
            visible = ((positions >= start) & (positions < end1)) | (
                (positions >= split) & (positions < end2)
            )
            scores[slot, :, pl.ds(page * PAGE_RECORDS, PAGE_RECORDS)] = jnp.where(
                visible, values, -jnp.inf
            )

        def qk(fragment):
            words = (
                jnp.concatenate(
                    [
                        key_rows((fragment * mxu + offset) // PAGE_RECORDS * fields, 32)
                        for offset in range(0, mxu, PAGE_RECORDS)
                    ]
                )
                if native_packing and fields == 33
                else key_rows(
                    fragment * mxu // PAGE_RECORDS * fields,
                    mxu // PAGE_RECORDS * fields,
                )
            )
            if lane_repack == "butterfly":
                # Exchange byte groups across token lanes, then perform one
                # final lane permutation. No narrow-column reshape is needed.
                channels = jnp.arange(dim, dtype=jnp.uint32)[None, :]
                native = words.astype(jnp.uint32)
                for bit, shift, mask in ((32, 8, 0x00FF00FF), (64, 16, 0xFFFF)):
                    even = (channels & bit) == 0
                    neighbor = pltpu.roll(native, bit, axis=1)
                    if bit == 32:
                        neighbor = jnp.where(
                            even, pltpu.roll(native, dim - bit, axis=1), neighbor
                        )
                    local_shift = jnp.where(even, shift, 0).astype(jnp.uint32)
                    delta = (
                        (native >> local_shift) ^ (neighbor >> (shift - local_shift))
                    ) & mask
                    native ^= delta << local_shift
                indices = jnp.broadcast_to(
                    (channels % 4) * 32 + channels // 4, words.shape
                )
                words = jnp.take_along_axis(
                    native, indices, axis=1, mode="promise_in_bounds"
                )
            elif lane_repack:
                # Four tokens occupy one 128-word row. Repack only their
                # 4x4 byte groups; keep rows and the 128-lane shape intact.
                channels = jnp.arange(dim, dtype=jnp.uint32)[None, :]
                shift = (channels % 4) * 8
                native = jnp.zeros_like(words, dtype=jnp.uint32)
                parts = []
                for token in range(4):
                    indices = jnp.broadcast_to(channels // 4 + token * 32, words.shape)
                    selected = jnp.take_along_axis(
                        words.astype(jnp.uint32),
                        indices,
                        axis=1,
                        mode="promise_in_bounds",
                    )
                    if lane_repack == "pack":
                        parts.append(selected >> shift)
                    else:
                        native |= ((selected >> shift) & 255) << (token * 8)
                words = (
                    pltpu.pack_elementwise(parts, packed_dtype=jnp.uint8)
                    if lane_repack == "pack"
                    else native
                )
            if native_packing or lane_repack:
                rhs = pltpu.bitcast(words, q.dtype)
                logits = lax.dot_general(
                    lhs,
                    rhs,
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))
            if byte_transpose:
                # Keep the transpose in packed FP8 instead of expanding a
                # 32-column int32 tensor into 128-lane register tiles.
                rhs = pltpu.bitcast(words, q.dtype)
                rhs = rhs.reshape(mxu // 4, 4, 4, 32).transpose(3, 1, 0, 2)
                rhs = rhs.reshape(128, mxu)
            else:
                words = words.reshape(mxu, fields)[:, :32]
                rhs = pltpu.bitcast(words.T, q.dtype)
            logits = lax.dot(lhs, rhs, preferred_element_type=jnp.float32)
            return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

        def emit(count, base=0):
            pending = {}
            for stage in range(count + config.issue_depth):
                if stage < count:
                    pending[stage] = qk(base + stage)
                if stage >= config.issue_depth:
                    fragment = stage - config.issue_depth
                    values = _reduce_heads(
                        pending.pop(fragment) * head_weights,
                        selector,
                        sublanes=info.num_sublanes,
                        use_mxu=stage < count and bool(fragment % 2),
                    )
                    for offset in range(0, mxu, PAGE_RECORDS):
                        page = ((base + fragment) * mxu + offset) // PAGE_RECORDS
                        scale_words = (
                            key_rows(page * fields + 32, 1)
                            if native_packing and fields == 33
                            else key_rows(page * fields, fields).reshape(
                                PAGE_RECORDS, fields
                            )[:, 32][None, :]
                            if fields == 33
                            else keys[1][step % config.key_buffers, pl.ds(page, 1)]
                        )
                        scale = lax.bitcast_convert_type(scale_words, jnp.float32)
                        store_page(
                            page, values[:, offset : offset + PAGE_RECORDS] * scale
                        )

        if loop_tile:
            fragment_count = min(loop_tile, config.key_tile) // mxu

            @pl.loop(0, pl.cdiv(limit(query), fragment_count * mxu))
            def score_tile(tile):
                emit(fragment_count, tile * fragment_count)
        else:
            prefix = min(config.key_tile, PACKED_FP8_PREPARED_PREFIX_TOKENS) // mxu
            total = config.key_tile // mxu
            branches = tuple(
                partial(emit, min(end, total))
                for end in range(prefix, total + prefix, prefix)
            )
            pl.when(limit(query) > 0)(
                lambda: lax.switch(pl.cdiv(limit(query), prefix * mxu) - 1, branches)
            )
        # One contiguous load/rotation after MXU issue avoids carrying a
        # partial output page through every static dot-product fragment.
        start = windows[query_rows[1], 0]
        aligned_start = start // PAGE_RECORDS * PAGE_RECORDS
        window = scores[
            slot,
            1:2,
            pl.ds(pl.multiple_of(aligned_start, PAGE_RECORDS), width + PAGE_RECORDS),
        ]
        shift = PAGE_RECORDS - start % PAGE_RECORDS
        rotated = pltpu.roll(window.reshape(-1, PAGE_RECORDS), shift, axis=1)
        scores[slot, 1:2, pl.ds(0, width)] = jnp.where(
            lanes[None, :] < shift, rotated[:-1], rotated[1:]
        ).reshape(1, width)
        _dma_rows(
            output(slot),
            out,
            write_sem.at[slot],
            query.count,
            src_start=query.start - offsets[query.page_row],
            dst_start=query.start - begin,
        )
        counts[slot] = query.count
        return following, lookahead, next_query(lookahead), step + 1

    zero = jnp.int32(0)
    first = next_query(QueryTile(begin, zero, zero))
    for slot in range(2):
        counts[slot] = zero
    second = next_query(first)
    load_query(first, zero)
    load_keys(first, zero)
    load_keys(second, jnp.int32(1))
    lax.while_loop(
        lambda state: state[0].start < begin + out.shape[0],
        body,
        (first, second, next_query(second), zero),
    )
    release(0)
    release(1)


@jax.jit(
    static_argnames=(
        "output_width",
        "fields",
        "query_count",
        "issue_depth",
        "loop_tile",
        "byte_transpose",
        "native_packing",
        "lane_repack",
    )
)
def token_major_mqa_logits(
    q: jax.Array,
    pools: tuple[jax.Array, ...],
    weights: jax.Array,
    windows: jax.Array,
    *,
    query_start_loc: jax.Array,
    output_width: int,
    fields: int,
    query_start: jax.Array | int = 0,
    query_count: int | None = None,
    issue_depth: int = 16,
    loop_tile: int = 0,
    byte_transpose: bool = False,
    native_packing: bool = False,
    lane_repack: str = "",
) -> jax.Array:
    """Identical pair-window scores for TM33 or TM32 plus separate scales."""
    rows = q.shape[0] if query_count is None else query_count
    info = cast(pltpu.TpuInfo, pltpu.get_tpu_info())
    groups = query_start_loc.shape[0] - 1
    capacity = pools[0].shape[0] // groups // fields * PAGE_RECORDS
    assert fields in (32, 33) and len(pools) == (2 if fields == 32 else 1)
    assert not byte_transpose or fields == 32
    assert not lane_repack or fields == 32
    assert lane_repack in ("", "direct", "butterfly", "pack")
    assert capacity % info.mxu_column_size == 0
    config = replace(
        PACKED_FP8_PREPARED_MQA_CONFIG, key_tile=capacity, issue_depth=issue_depth
    )
    q_tile, heads, dim = config.query_tile, q.shape[1], q.shape[2]
    widths = (fields,) if fields == 33 else (32, 1)
    scratch = (
        pltpu.VMEM((2 * q_tile, heads, dim), q.dtype),
        tuple(
            pltpu.VMEM(
                (config.key_buffers, capacity // PAGE_RECORDS * n, PAGE_RECORDS),
                jnp.int32,
            )
            for n in widths
        ),
        pltpu.VMEM((2, q_tile, capacity + PAGE_RECORDS), jnp.float32),
        pltpu.SMEM((2,), jnp.int32),
        pltpu.SemaphoreType.DMA((config.key_buffers, len(widths))),
        pltpu.SemaphoreType.DMA((2, q_tile)),
        pltpu.SemaphoreType.DMA((2,)),
    )
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    return pl.pallas_call(
        partial(
            _kernel,
            config=config,
            info=info,
            fields=fields,
            loop_tile=loop_tile,
            byte_transpose=byte_transpose,
            native_packing=native_packing,
            lane_repack=lane_repack,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, output_width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((hbm, vmem, tuple(hbm for _ in pools)),),
            out_specs=hbm,
            scratch_shapes=(scratch,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        name=f"token_major_{fields}_mqa_logits",
    )(
        query_start_loc,
        windows,
        jnp.asarray(query_start, jnp.int32),
        (q, weights, pools),
    )


def gather_token_major(
    pools: tuple[tuple[jax.Array, ...], ...],
    physical_block_ids: jax.Array,
    offsets: jax.Array,
    *,
    fields: int,
    block: int = 8,
    group: int = 64,
    buffers: int = 4,
    repack: bool = False,
) -> tuple[jax.Array, ...]:
    """Gather contiguous token records, retaining the production DMA pipeline.

    Pools contain either TM33 words, or TM32 words and a separate scale stream.
    Banks split at token boundaries. This experiment uses tightly packed pools.
    """
    from jax.experimental.pallas import tpu_sc as plsc

    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    widths = (block * fields,) if fields == 33 else (block * 32, block)
    banks = len(pools[0])
    count = physical_block_ids.size
    per_unit = pl.cdiv(count, units * group) * group
    steps = per_unit // group
    scratch_words = buffers * (group * (sum(widths) + banks) + lanes) + lanes
    if repack:
        assert fields == 33 and group * block % PAGE_RECORDS == 0
        scratch_words += buffers * group * widths[0]
    metadata_rows = min(
        pl.cdiv(per_unit, group * lanes) * group * lanes,
        (info.vmem_capacity_bytes // 4 - scratch_words - 4096)
        // (group * lanes)
        * group
        * lanes,
    )
    assert metadata_rows > 0 and group % lanes == 0
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
    rotate = physical_block_ids.shape[1] % (units * group) == 0
    steps_per_group = max(1, physical_block_ids.shape[1] // (units * group))
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=tuple(
            jax.ShapeDtypeStruct(
                (units * per_unit * n // PAGE_RECORDS, PAGE_RECORDS), jnp.int32
            )
            for n in widths
        ),
        scratch_types=(
            pltpu.VMEM((meta_steps, group), jnp.int32),
            pltpu.VMEM((meta_steps,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.VMEM((buffers, lanes), jnp.int32),
            tuple(pltpu.VMEM((buffers, group, n), jnp.int32) for n in widths),
            (pltpu.VMEM((buffers, group * widths[0]), jnp.int32),) if repack else (),
            pltpu.VMEM((buffers, banks, group), jnp.int32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((buffers, len(widths), banks)),
            pltpu.SemaphoreType.DMA((buffers, len(widths))),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False, needs_layout_passes=False
        ),
        mesh=mesh,
        name=f"token_major_{fields}_gather",
    )
    def kernel(pool, physical_ref, control_ref, *refs):
        outputs = refs[: len(widths)]
        (
            selected,
            metadata_ids,
            ctrl,
            flags,
            words,
            packed,
            read_ids,
            meta_sem,
            read_sem,
            write_sem,
        ) = refs[len(widths) :]
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        pltpu.async_copy(control_ref, ctrl, meta_sem).wait()
        active = ctrl[...][0]

        def destination(step):
            return step * units + (
                (unit + step // steps_per_group) % units if rotate else unit
            )

        def load_metadata(start):
            for i in range(0, meta_steps, lanes):
                row = start + i + jnp.arange(lanes, dtype=jnp.int32)
                metadata_ids[pl.ds(i, lanes)] = jnp.where(
                    row < steps, destination(row), -1
                )
            pltpu.async_copy(
                physical_ref.at[plsc.Indices(metadata_ids, ignored_value=-1)],
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
                for i in range(0, group, lanes):
                    chosen = selected.reshape(-1)[pl.ds(index * group + i, lanes)]
                    any_valid |= chosen >= 0
                    bank_base = 0
                    for bank, ref in enumerate(pool[0]):
                        bank_blocks = ref.shape[0] * ref.shape[1] // widths[0]
                        valid = (chosen >= bank_base) & (
                            chosen < bank_base + bank_blocks
                        )
                        read_ids[slot, bank, pl.ds(i, lanes)] = jnp.where(
                            valid, chosen - bank_base, -1
                        )
                        bank_base += bank_blocks
                flags[slot, :] = (
                    plsc.all_reduce_population_count(any_valid) != 0
                ).astype(jnp.int32)

            @pl.when(live(step))
            def transfer():
                for stream, n in enumerate(widths):
                    for bank, ref in enumerate(pool[stream]):
                        copy = pltpu.make_async_copy(
                            ref.reshape(-1, n).at[
                                plsc.Indices(read_ids.at[slot, bank], ignored_value=-1)
                            ],
                            words[stream].at[slot],
                            read_sem.at[slot, stream, bank],
                        )
                        (copy.wait if wait else copy.start)()

        def write(step, wait=False):
            slot = step % buffers

            @pl.when(live(step))
            def transfer():
                if repack:
                    if not wait:

                        @plsc.parallel_loop(0, group * widths[0], lanes, unroll=8)
                        def transpose(i):
                            position = i + jnp.arange(lanes)
                            token = (
                                position // (33 * PAGE_RECORDS) * PAGE_RECORDS
                                + position % PAGE_RECORDS
                            )
                            field = position // PAGE_RECORDS % 33
                            packed[0][slot, pl.ds(i, lanes)] = plsc.load_gather(
                                words[0].at[slot].reshape(-1), [token * 33 + field]
                            )

                    target = (
                        outputs[0]
                        .reshape(-1)
                        .at[
                            pl.ds(
                                destination(step) * group * widths[0], group * widths[0]
                            )
                        ]
                    )
                    copy = pltpu.make_async_copy(
                        packed[0].at[slot], target, write_sem.at[slot, 0]
                    )
                    (copy.wait if wait else copy.start)()
                    return
                for stream, n in enumerate(widths):
                    target = (
                        outputs[stream]
                        .reshape(-1, n)
                        .at[pl.ds(destination(step) * group, group)]
                    )
                    copy = pltpu.make_async_copy(
                        words[stream].at[slot], target, write_sem.at[slot, stream]
                    )
                    (copy.wait if wait else copy.start)()

        for step in range(min(buffers - 1, steps)):
            pl.when(step < active)(
                lambda step=step: read(jnp.int32(step), jnp.int32(step))
            )

        @pl.loop(0, active)
        def body(step):
            read(step, step % buffers, wait=True)
            write(step)

            @pl.when(step + buffers - 1 < active)
            def prefetch():
                future = step + buffers - 1
                pl.when(future >= buffers)(lambda: write(future - buffers, wait=True))
                read(future, future % buffers)

        for last in range(buffers):
            pl.when(last < active)(
                lambda last=last: write(active - last - 1, wait=True)
            )

    result = kernel(pools, ids, control)
    return tuple(value[: count * n // PAGE_RECORDS] for value, n in zip(result, widths))
