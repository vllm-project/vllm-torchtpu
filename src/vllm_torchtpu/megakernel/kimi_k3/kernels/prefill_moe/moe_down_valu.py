# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions of this file are modified copies of code Copyright 2026 Google LLC,
# licensed under the Apache License, Version 2.0, from
# vllm_torchtpu/kernels/megablox/gmm_v2.py.
"""MXFP4 down, weighted token accumulation, and pipelined reduce-scatter.

While the MXU produces tile n, VALU accumulates tile n-1 in packed token
layout. Completed column-block subtotals feed the ring directly from VMEM;
the previous block's ring advances between tiles of the next down block.
Products and local subtotals round to BF16; token weighting uses FP32 and
the ring uses ``ring_dtype``.
"""

import contextlib
import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k


def ring_step(
    tables_ref,
    x_ref,
    o_ref,
    local_sems,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    local_buf,
    payload,
    received,
    *,
    streams,
    buffers,
    step,
):
    tile = 0
    rank = lax.axis_index("k3_sp")
    descriptors = []
    reads = []
    positions = [tables_ref[f, 0, rank] for f in range(3)]

    def ring_coords(pos, family):
        peer = tables_ref[family, 1, pos]
        return (peer,)

    @pl.when(step == 0)
    def enter():
        peers = [
            ring_coords((positions[f] + direction) % 32, jnp.int32(f))
            for f in range(3)
            for direction in (-1, 1)
        ]
        sem = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

    for stream, region in enumerate(streams):
        begin, size = region[:2]
        col_begin, width = region[2:] if len(region) == 4 else (0, o_ref.shape[1])
        cols = pl.ds(col_begin, width)
        family = stream // 2 % 3
        direction = -1 if stream % 2 == 0 else 1
        position = positions[family]
        target = ring_coords((position + direction) % 32, jnp.int32(family))
        source = ring_coords((position - direction) % 32, jnp.int32(family))
        owner_mesh = ring_coords(
            (position - direction * (step + 1)) % 32, jnp.int32(family)
        )[0]
        owner = tables_ref[family, 2, owner_mesh]
        rows = pl.ds(begin, size)
        source_rows = pl.ds(
            owner * o_ref.shape[0] + tile * payload.shape[0] + begin, size
        )
        reads.append(None)
        descriptors.append((target, source, rows, cols, source_rows))
    for stream, (target, source, rows, cols, source_rows) in enumerate(descriptors):

        def local_value(rows=rows, cols=cols, source_rows=source_rows):
            return (
                x_ref.reshape(x_ref.shape[0], o_ref.shape[1])
                .at[source_rows, cols][...]
                .astype(payload.dtype)
            )
            return local_buf.at[rows, cols][...].astype(payload.dtype)

        def remote(index, rows=rows, cols=cols, stream=stream, target=target):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[stream],
                recv_sems.at[index % buffers, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step > 0)
        def wait(remote=remote):
            remote(step - 1).wait()

        @pl.when(step == 0)
        def seed(rows=rows, cols=cols, local_value=local_value):
            payload.at[rows, cols][...] = local_value()

        @pl.when(step > 0)
        def add(
            rows=rows, cols=cols, stream=stream, source=source, local_value=local_value
        ):
            payload.at[rows, cols][...] = (
                local_value() + received.at[(step - 1) % buffers, rows, cols][...]
            )
            pl.semaphore_signal(
                credits.at[stream],
                1,
                device_id=source,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step < 31)
        def forward(remote=remote, stream=stream):
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[stream], 1)

            remote(step).start()

        @pl.when(step == 31)
        def finish(rows=rows, cols=cols, stream=stream):
            pl.semaphore_wait(credits.at[stream], buffers)

    @pl.when(step == 31)
    def store_complete_shard():
        output = pltpu.make_async_copy(payload, o_ref, store_sem)
        output.start()
        output.wait()


def down_valu(
    x,
    rhs,
    scales,
    counts,
    destinations,
    probabilities,
    tables,
    *,
    tokens,
    columns=1792,
    tile=128,
    unroll=8,
    layer=None,
    stream_metadata=False,
    metadata_cache_rows=0,
    metadata_row_capacity=0,
    ring_dtype=jnp.float32,
):
    """Complete BF16 products are immediately combined with FP32 routing weights.

    Destination metadata follows expert-major product rows. There is one writer
    per local token accumulator. Columns share one bounded accumulator; only
    complete local subtotals are rounded to BF16 for the ring, whose running
    sum and reduced output are ``ring_dtype``.
    """
    assert tile in (64, 128, 256) and unroll > 0
    assert tokens in (4096, 8192)
    rows, width = (metadata_row_capacity or x.shape[0], rhs.shape[-1])
    assert rows <= x.shape[0]
    tm, tn, localrows = (tile, columns, tokens // 32)
    assert width % tn == 0
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)
    offset = jnp.zeros((1,), jnp.int32)
    family = rhs.ndim == 4
    if family != (layer is not None):
        raise ValueError("Resident weights require a layer index")
    if layer is None:
        layer = offset
    cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, x.shape[-1]), x.dtype),
        jax.ShapeDtypeStruct(rhs.shape[1:] if family else rhs.shape, rhs.dtype),
        jax.ShapeDtypeStruct(
            scales.shape[1:] if family else scales.shape, scales.dtype
        ),
        None,
        counts,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=False,
        zero_initialize=False,
        fuse_act=None,
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    cfg = dataclasses.replace(
        cfg, tiles=dataclasses.replace(cfg.tiles, tile_m=tm, tile_n=tn)
    )
    sub = cfg.dims.size_lhs_sublane
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tm) - 1
    regions = k._regions(localrows, tn)
    streams = len(regions)
    store_tile = 128
    if metadata_cache_rows:
        assert stream_metadata and metadata_cache_rows % 128 == 0
    # Cache adjacent 8K metadata chunks across product and column tiles.
    route_chunk = min(rows, metadata_cache_rows or 8192)
    route_span = route_chunk + tm

    def scope(name):
        enabled = False
        return jax.named_scope(name) if enabled else contextlib.nullcontext()

    def kernel(
        counts,
        offset,
        layer,
        tables,
        destinations,
        probabilities,
        x,
        w,
        out,
        subtotals,
        partial,
        acc,
        metadata,
        y,
        packed,
        sums,
        cast,
        sum_sem,
        loads,
        sends,
        receives,
        credits,
        stores,
        localbuf,
        payload,
        rx,
        state,
        subtotal_vmem,
        dest_tiles,
        prob_tiles,
        route_sem,
        route_tags,
    ):
        if family:
            w = dataclasses.replace(
                w, weight=w.weight.at[layer[0]], scale=w.scale.at[layer[0]]
            )
        w = dataclasses.replace(w, weight=w.weight.bitcast(jnp.float4_e2m1fn))
        active_ng = k.fill_metadata(counts, offset, metadata, cfgs=cfg)

        @pl.when(active_ng == 0)
        def empty_metadata():
            metadata.gm_id_to_group_id[0] = 0
            metadata.gm_id_to_m_offset[0] = 0
            metadata.gm_id_to_m_offset[1] = 0

        # Empty ranks retain the same collective call sites as active ranks.
        # A separate empty-rank ring branch can allocate different semaphores.
        ng = jnp.maximum(active_ng, 1)
        (xs, ws), _ = k.generate_block_specs(metadata, cfg)
        state[0] = 0
        if metadata_cache_rows:
            route_tags[0] = -1
            route_tags[1] = -1

        def advance(stop, block):
            with scope("ring_progress"):

                @pl.loop(state[0], stop)
                def hop(step):
                    ring_step(
                        tables,
                        subtotal_vmem,
                        out.at[:, pl.ds(block * tn, tn)],
                        loads,
                        sends,
                        receives,
                        credits,
                        stores,
                        localbuf,
                        payload,
                        rx,
                        streams=regions,
                        buffers=2,
                        step=step,
                    )

            state[0] = stop

        def emit_subtotals(n):
            with scope("subtotal_cast_store"):

                @pl.loop(0, tokens // store_tile)
                def block(b):
                    cast[...] = (
                        sums[pl.ds(b * store_tile, store_tile)]
                        .reshape(store_tile, tn)
                        .astype(jnp.bfloat16)
                    )
                    subtotal_vmem.at[pl.ds(b * store_tile, store_tile)][...] = cast[...]

        def route(tile_g, i):
            if stream_metadata:
                chunk = metadata.gm_id_to_m_offset[tile_g] // route_chunk
                begin = chunk * route_chunk
                return (
                    dest_tiles[chunk % 2, i - begin],
                    prob_tiles[chunk % 2, i - begin],
                )
            return (destinations[i], probabilities[i])

        def consume(tile_g, source, first=0):
            start = metadata.gm_id_to_m_offset[tile_g]
            end = metadata.gm_id_to_m_offset[tile_g + 1]
            begin = start // sub * sub
            start = jnp.minimum(start + first, end)

            def update(i):
                dest, weight = route(tile_g, i)
                value = source.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
                target = sums.at[pl.ds(dest, 1)]
                target[...] = target[...] + value * weight

            with scope("valu_accumulate"):
                full = (end - start) // unroll

                @pl.loop(0, full)
                def chunk(b):
                    for u in range(unroll):
                        update(start + b * unroll + u)

                @pl.loop(start + full * unroll, end)
                def tail(i):
                    update(i)

        def inner(x, w, partial, acc, metadata):
            g, n = (pl.program_id(1), pl.program_id(0))
            if stream_metadata:
                chunk = metadata.gm_id_to_m_offset[g] // route_chunk
                previous_chunk = (
                    metadata.gm_id_to_m_offset[jnp.maximum(g - 1, 0)] // route_chunk
                )
                missing = (
                    route_tags[chunk % 2] != chunk
                    if metadata_cache_rows
                    else (g == 0) | (chunk != previous_chunk)
                )

                @pl.when(missing)
                def load_routes():
                    begin = chunk * route_chunk
                    copies = (
                        pltpu.make_async_copy(
                            destinations.at[pl.ds(begin, route_span)],
                            dest_tiles.at[chunk % 2],
                            route_sem,
                        ),
                        pltpu.make_async_copy(
                            probabilities.at[pl.ds(begin, route_span)],
                            prob_tiles.at[chunk % 2],
                            route_sem,
                        ),
                    )
                    for cp in copies:
                        cp.start()
                    for cp in copies:
                        cp.wait()
                    if metadata_cache_rows:
                        route_tags[chunk % 2] = chunk

            @pl.when(g == 0)
            def clear():
                with scope("accumulator_clear"):
                    sums[...] = jnp.zeros(sums.shape, jnp.float32)

            @pl.when(n > 0)
            def progress():
                advance(jnp.minimum(32, (g + 1) * 32 // ng), n - 1)

            current = packed.at[g % 2]
            previous = packed.at[(g + 1) % 2]
            previous_g = jnp.maximum(g - 1, 0)
            panel_count = pl.cdiv(tn, pltpu.get_tpu_info().mxu_column_size)

            # These callbacks issue VALU work between MXU panel issue/use.
            # Ping-pong output banks keep the previous tile live throughout.
            def overlap_work(panel):
                start = metadata.gm_id_to_m_offset[previous_g]
                end = metadata.gm_id_to_m_offset[previous_g + 1]
                begin = start // sub * sub
                with scope("previous_valu_chunk"):
                    for row in range(unroll):
                        offset = panel * unroll + row
                        i = jnp.minimum(start + offset, end - 1)
                        dest, prob = route(previous_g, i)
                        weight = jnp.where((g > 0) & (start + offset < end), prob, 0)
                        value = previous.at[pl.ds(i - begin, 1)][...].astype(
                            jnp.float32
                        )
                        target = sums.at[pl.ds(dest, 1)]
                        target[...] = target[...] + value * weight

            @pl.when(g == 0)
            def initialize_previous():
                previous[...] = jnp.zeros(previous.shape, previous.dtype)

            @pl.when(active_ng > 0)
            def product():
                with scope("down_compute"):
                    packed_inner_kernel(
                        x,
                        w,
                        current,
                        partial,
                        acc,
                        metadata,
                        cfgs=cfg,
                        overlap_work=overlap_work,
                    )

            @pl.when(g > 0)
            def previous_tail():
                consume(previous_g, previous, panel_count * unroll)

            @pl.when(g == ng - 1)
            def finish():
                consume(g, current)

                @pl.when(n > 0)
                def previous():
                    with scope("ring_previous_drain"):
                        advance(jnp.int32(32), n - 1)

                emit_subtotals(n)
                state[0] = 0

                @pl.when(n == width // tn - 1)
                def last():
                    with scope("ring_final_drain"):
                        advance(jnp.int32(32), n)

        pltpu.emit_pipeline(
            inner, grid=(width // tn, ng, 1), in_specs=(xs, ws), out_specs=()
        )(x.reshape(-1, sub, x.shape[-1]), w, scratches=(partial, acc, metadata))

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((localrows, width), ring_dtype),
            jax.ShapeDtypeStruct((tokens, tn), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4 if stream_metadata else 6,
            in_specs=((hbm, hbm) if stream_metadata else ())
            + (hbm, k.WeightsRef(weight=hbm, scale=hbm, bias=None)),
            out_specs=(hbm, hbm),
            scratch_shapes=(
                None,
                pltpu.VMEM((tm, tn), jnp.float32),
                k.MetadataRef(
                    gm_id_to_group_id=pltpu.SMEM((max_gm,), jnp.int32),
                    gm_id_to_m_offset=pltpu.SMEM((max_gm + 1,), jnp.int32),
                ),
                None,
                pltpu.VMEM((2,) + (tm, 2, tn // 2), x.dtype),
                pltpu.VMEM((tokens, 2, tn // 2), jnp.float32),
                pltpu.VMEM((store_tile, tn), x.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((2, streams)),
                pltpu.SemaphoreType.REGULAR((streams,)),
                pltpu.SemaphoreType.DMA,
                None,
                pltpu.VMEM((localrows, tn), ring_dtype),
                pltpu.VMEM((2, localrows, tn), ring_dtype),
                pltpu.SMEM((1,), jnp.int32),
                pltpu.VMEM((tokens, tn), x.dtype),
                pltpu.SMEM((2, route_span), jnp.int32) if stream_metadata else None,
                pltpu.SMEM((2, route_span), jnp.float32) if stream_metadata else None,
                pltpu.SemaphoreType.DMA if stream_metadata else None,
                pltpu.SMEM((2,), jnp.int32) if metadata_cache_rows else None,
            ),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=77, vmem_limit_bytes=limit),
        name=f"down_valu_{'spread'}_m{tm}_n{tn}_u{unroll}_v{int(True)}_b{int(False)}_p{int(True)}_f{int(False)}_h{int(False)}_g{1}_c{int(False)}_x{int(True)}_z{int(True)}_r{int(False)}_d{int(True)}_o{int(True)}_mc{metadata_cache_rows}",
    )
    return call(
        counts,
        offset,
        layer,
        jnp.asarray(tables, jnp.int32),
        jnp.pad(destinations, (0, route_span)) if stream_metadata else destinations,
        jnp.pad(probabilities, (0, route_span)) if stream_metadata else probabilities,
        x,
        k.WeightsRef(weight=rhs, scale=scales, bias=None),
    )[0]


def packed_inner_kernel(
    tiled_lhs_ref: jax.Array,
    tiled_rhs_ref: k.RhsRef,
    tiled_out_ref: jax.Array,
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: k.MetadataRef,
    *,
    cfgs: k.GmmConfigs,
    overlap_work=None,
):
    """Variant of ``prefill_megakernel.inner_kernel`` with a direct packed final store.

    Arithmetic and row bucketing follow k.inner_kernel. Each valid expert row
    is consumed immediately, so no adjacent-group sublane carry is needed.
    Output is [tile_m, 2, tile_n // 2]; partial_out_ref is deliberately unused.
    Products round to BF16 before the caller's FP32 routing weighting.
    """
    if overlap_work is not None:
        assert cfgs.dims.size_k == cfgs.tiles.tile_k
        assert cfgs.fuse_act is None and (not cfgs.rhs_cfgs.has_bias)
    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def _matmul(is_first_k_step: bool, is_last_k_step: bool, bucket_m: int):
        tpu_info = pltpu.get_tpu_info()
        mxu_size = tpu_info.mxu_column_size
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
        tiled_rhs = tiled_rhs_ref.get_weight()
        rhs_tile_n = tiled_rhs.shape[1]
        rhs_qbs = cfgs.rhs_cfgs.quant_block_size
        assert rhs_qbs is not None
        if cfgs.rhs_cfgs.should_dequantize_before_matmul:
            tiled_rhs_scale = tiled_rhs_ref.get_scale(replicate_size=rhs_qbs).astype(
                cfgs.lhs_cfgs.dtype
            )
            num_blocks = cfgs.num_quant_blocks_per_tile_k
            tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(
                num_blocks, rhs_qbs, rhs_tile_n
            )
            tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
            tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)
            rhs_qbs = cfgs.tiles.tile_k
        valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
        if is_last_k_step and valid_k != 0:
            mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
            tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)
        acc_list = []
        column_order = list(range(0, rhs_tile_n, mxu_size))

        def accept_fragment(acc_n, start_n, end_n):
            acc_list.append(acc_n)

        if not cfgs.lhs_cfgs.should_quantize:
            for start_n in column_order:
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n
                acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
                for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
                    end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)
                    block_acc = jnp.matmul(
                        tiled_lhs[:, start_k:end_k],
                        tiled_rhs[start_k:end_k, start_n:end_n],
                        preferred_element_type=jnp.float32,
                    ).astype(acc_ref.dtype)
                    if overlap_work is not None and end_k == cfgs.tiles.tile_k:
                        overlap_work(start_n // mxu_size)
                    if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                        b_id = start_k // rhs_qbs
                        rhs_scale_replicated = tiled_rhs_ref.get_scale(
                            replicate_size=bucket_m
                        )[b_id, :, start_n : start_n + col_size]
                        block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)
                    acc_n += block_acc
                accept_fragment(acc_n, start_n, end_n)
        else:
            lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
            q_block_size = cfgs.lhs_cfgs.quant_block_size
            assert q_block_size is not None
            if jnp.issubdtype(lhs_q_dtype, jnp.floating):
                dtype_max = float(jnp.finfo(lhs_q_dtype).max)
                preferred_element_type = jnp.float32
            else:
                dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
                preferred_element_type = jnp.int32
            for start_n in column_order:
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n
                acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
                for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)
                    block_lhs = tiled_lhs[:, start_k:end_k]
                    block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]
                    block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
                    block_scale = block_abs_max / dtype_max
                    block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
                    block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)
                    if not tpu_info.is_matmul_supported(lhs_q_dtype, block_rhs.dtype):
                        block_rhs = block_rhs.astype(lhs_q_dtype)
                    block_acc = jnp.matmul(
                        block_lhs_q,
                        block_rhs,
                        preferred_element_type=preferred_element_type,
                    ).astype(acc_ref.dtype)
                    if overlap_work is not None and end_k == cfgs.tiles.tile_k:
                        overlap_work(start_n // mxu_size)
                    block_acc *= block_scale.astype(acc_ref.dtype)
                    if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                        b_id = start_k // rhs_qbs
                        rhs_scale_replicated = tiled_rhs_ref.get_scale(
                            replicate_size=bucket_m
                        )[b_id, :, start_n : start_n + col_size]
                        block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)
                    acc_n += block_acc
                accept_fragment(acc_n, start_n, end_n)
        acc = jnp.concatenate(acc_list, axis=1)
        if not is_first_k_step:
            acc = jnp.pad(acc, ((cfgs.tiles.tile_m - bucket_m, 0), (0, 0)))
            acc += acc_ref[...]
        acc_m = acc.shape[0]
        if is_last_k_step:
            if cfgs.rhs_cfgs.has_bias:
                tiled_rhs_bias = tiled_rhs_ref.get_bias()
                acc += tiled_rhs_bias.astype(acc.dtype)
            acc = k.apply_act_fn(acc, cfgs.fuse_act)
            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
            acc_masked = jnp.where(mask, acc, 0)
            tiled_out_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype).reshape(
                acc_m, 2, cfgs.tiles.tile_n // 2
            )
        else:
            acc_ref[:acc_m] = acc

    def run_matmul_step(bucket_idx: int):
        bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)
        if overlap_work is not None:
            with jax.named_scope(f"bm{bucket_m}_fragment"):
                _matmul(is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m)
            return

        @jax.named_scope(f"bm{bucket_m}_first_last")
        def matmul_first_last():
            _matmul(is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_first")
        def matmul_first():
            _matmul(is_first_k_step=True, is_last_k_step=False, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_mid")
        def matmul_mid():
            _matmul(is_first_k_step=False, is_last_k_step=False, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_last")
        def matmul_last():
            _matmul(is_first_k_step=False, is_last_k_step=True, bucket_m=bucket_m)

        num_k = pl.num_programs(2)
        k_id = pl.program_id(2)
        is_first_k_step = k_id == 0
        is_last_k_step = k_id == num_k - 1
        if bucket_m == cfgs.tiles.tile_m:
            lax.cond(
                is_first_k_step,
                lambda: lax.cond(is_last_k_step, matmul_first_last, matmul_first),
                lambda: lax.cond(is_last_k_step, matmul_last, matmul_mid),
            )
        else:
            lax.cond(is_first_k_step, matmul_first_last, matmul_last)

    branches = []
    for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base):
        branches.append(functools.partial(run_matmul_step, bucket_idx=bucket_idx))
    bucket_idx = m_end_local // cfgs.tiles.bucket_base
    lax.switch(bucket_idx, branches)


def down_valu_compact(
    x,
    rhs,
    scales,
    counts,
    destinations,
    probabilities,
    tables,
    *,
    metadata_prefix_rows,
    slice_input=True,
    global_route_max=None,
    **kwargs,
):
    """Uniform compact-prefix fast path with an exact, uncapped overflow path.

    All ranks must select the same collective-containing branch. The caller
    supplies the maximum derived from globally replicated route counts to
    avoid an additional collective. The full activation buffer stays in HBM;
    only the scalar metadata prefix is bounded. Overflow never drops routes.
    """
    capacity = metadata_prefix_rows
    peak = (
        lax.pmax(counts.sum(), "k3_sp")
        if global_route_max is None
        else global_route_max
    )
    fits = peak <= capacity
    compact_args = dict(kwargs, stream_metadata=False, metadata_cache_rows=0)
    overflow_args = dict(kwargs, stream_metadata=True, metadata_cache_rows=8192)
    return lax.cond(
        fits,
        lambda: down_valu(
            x[:capacity] if slice_input else x,
            rhs,
            scales,
            counts,
            destinations[:capacity],
            probabilities[:capacity],
            tables,
            metadata_row_capacity=capacity,
            **compact_args,
        ),
        lambda: down_valu(
            x, rhs, scales, counts, destinations, probabilities, tables, **overflow_args
        ),
    )
