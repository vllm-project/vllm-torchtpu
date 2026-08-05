import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
    config, memory_ref)


def call_kernel_pipeline(
    row_partition_id: jax.Array,
    scalar_ref: memory_ref.IndexRef,
    scratch_ref: memory_ref.ScratchRef,
    in_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    col_start: jax.Array,
    cfg: config.Config,
):
    num_rows_per_row_partition = scratch_ref.num_rows_per_row_partition_vmem[
        ...]
    num_rows_current_row_partition = jnp.array(0, jnp.int32)
    for i in range(cfg.num_row_partitions):
        num_rows_current_row_partition = jnp.where(
            row_partition_id == i,
            num_rows_per_row_partition[i],
            num_rows_current_row_partition,
        )
    num_row_blocks = pl.cdiv(num_rows_current_row_partition,
                             cfg.row_chunk_size)

    # Sentinel for the cross-block reduction carry (no previous group).
    scratch_ref.prev_dst_row_smem[0] = -1

    row_gather_specs = _row_gather_specs(scratch_ref.sorted_by_validity_vmem,
                                         cfg)

    row_pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            _row_kernel,
            cfg=cfg,
            scratch_ref=scratch_ref,
            in_hbm_ref=in_hbm_ref,
            out_hbm_ref=out_hbm_ref,
            num_rows_current_row_partition=num_rows_current_row_partition,
            col_start=col_start,
        ),
        grid=(num_row_blocks, ),
        in_specs=(row_gather_specs, row_gather_specs),
    )

    row_pipeline_fn(
        ((scalar_ref.indices, ) * cfg.num_row_subchunks),
        ((topk_weights_hbm_ref, ) * cfg.num_row_subchunks),
    )


def _pack_scalars_to_vector(scalar_list: list[jax.Array]) -> jax.Array:
    """Pack list of scalar values into a single VMEM lane."""
    num_lanes = len(scalar_list)

    idx_vec = jax.lax.broadcasted_iota(jnp.int32, (num_lanes, ), 0)
    vec = jnp.zeros((num_lanes, ), jnp.int32)
    for i in range(num_lanes):
        vec = jnp.where(idx_vec == i, scalar_list[i], vec)
    return vec


def _row_gather_specs(sorted_by_validity_vmem: jax.Ref,
                      cfg: config.Config) -> tuple[pl.BlockSpec, ...]:
    """Indirect BlockSpec gathering rows of a 1-D input."""

    num_simd_lanes = cfg.sc_info.num_lanes

    def row_index_map(r: int | jax.Array, *, offset: int) -> jax.Array:
        start = r * cfg.row_chunk_size + offset * num_simd_lanes
        return sorted_by_validity_vmem[pl.ds(start, num_simd_lanes)]

    return tuple([
        pl.BlockSpec(
            (pl.Indirect(num_simd_lanes), ),
            functools.partial(row_index_map, offset=offset),
        ) for offset in range(cfg.num_row_subchunks)
    ])


def _col_gather_spec(
    src_indices_vmem: jax.Ref,
    col_start: jax.Array,
    cfg: config.Config,
) -> pl.BlockSpec:
    """Indirect BlockSpec gathering columns of a 2-D input."""

    num_simd_lanes = cfg.sc_info.num_lanes

    def col_index_map(s: int | jax.Array,
                      c: int | jax.Array) -> tuple[jax.Array, jax.Array]:
        row = jnp.bitwise_right_shift(
            src_indices_vmem[pl.ds(s * num_simd_lanes, num_simd_lanes)],
            cfg.row_shift,
        )
        col = col_start // cfg.col_chunk_size + c
        return (row, col)

    return pl.BlockSpec(
        (pl.Indirect(num_simd_lanes), cfg.col_chunk_size),
        col_index_map,
    )


def _row_kernel(
    src_indices_refs: tuple[jax.Ref, ...],
    topk_weights_refs: tuple[jax.Ref, ...],
    *,
    scratch_ref: memory_ref.ScratchRef,
    in_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    num_rows_current_row_partition: jax.Array,
    col_start: jax.Array,
    cfg: config.Config,
):
    # The SparseCore indirect DMA requires 32-bit elements, so x is gathered
    # through a uint32 reinterpretation. bfloat16 packs two source rows per
    # uint32 row (row index >> 1); float32 is 1:1 (row index unchanged).
    in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)

    num_simd_lanes = cfg.sc_info.num_lanes

    row_block_id = pl.program_id(0)

    # Destination output row of each source row in this block.
    dst_indices_list = []
    for s in range(cfg.num_row_subchunks):
        start = row_block_id * cfg.row_chunk_size + s * num_simd_lanes
        gather_dst = scratch_ref.sorted_by_validity_vmem[pl.ds(
            start, num_simd_lanes)]
        dst_indices_list.append(gather_dst // cfg.reduce_group_size)

    # Stage the gathered indices/weights and the destinations in VMEM.
    for s in range(cfg.num_row_subchunks):
        sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
        scratch_ref.src_indices_vmem[sub] = src_indices_refs[s][...]
        scratch_ref.dst_indices_vmem[sub] = dst_indices_list[s]

        tw = topk_weights_refs[s][...]
        if cfg.topk_dtype == jnp.bfloat16:
            tw_f32 = plsc.bitcast(jnp.bitwise_left_shift(tw, 16), jnp.float32)
        else:
            tw_f32 = plsc.bitcast(tw, jnp.float32)
        scratch_ref.tw_f32_vmem[sub] = tw_f32

    # For each sub-chunk, the destination of the row just before it -- the
    # seed for the segmented reduction's "same group as previous row" test.
    for s in range(cfg.num_row_subchunks):
        if s == 0:
            prev_dst = scratch_ref.prev_dst_row_smem[0]
        else:
            prev_dst = dst_indices_list[s - 1][num_simd_lanes - 1]
        scratch_ref.prev_dst_val_vmem[pl.ds(
            s * num_simd_lanes,
            num_simd_lanes)] = (jnp.broadcast_to(prev_dst, (num_simd_lanes, )))

    # For each source row, find the VMEM row that will hold its group's fully
    # reduced value -- the last row of the group within this block. Scanning
    # backwards, a row inherits its successor's merge target when they share
    # a destination, otherwise it is its own target.
    src_row_idx_in_vmem = []
    row_valid_vec = []
    for row_vmem_idx in reversed(range(cfg.row_chunk_size)):
        global_row_idx = row_block_id * cfg.row_chunk_size + row_vmem_idx
        row_valid_vec.append(global_row_idx < num_rows_current_row_partition)
        if row_vmem_idx == cfg.row_chunk_size - 1:
            src_row_idx_in_vmem.append(row_vmem_idx)
        else:
            quot, rem = divmod(row_vmem_idx, num_simd_lanes)
            quot_next, rem_next = divmod(row_vmem_idx + 1, num_simd_lanes)
            same_group_as_next = jnp.logical_and(
                row_valid_vec[-2],
                dst_indices_list[quot][rem] == dst_indices_list[quot_next]
                [rem_next],
            ).astype(jnp.int32)
            src_row_idx_in_vmem.append(same_group_as_next *
                                       src_row_idx_in_vmem[-1] +
                                       (1 - same_group_as_next) * row_vmem_idx)
    src_row_idx_in_vmem.reverse()
    row_valid_vec.reverse()

    # Per source row, the (VMEM source row, HBM destination row) of its
    # scatter. Rows whose group is not yet fully reduced in this sub-chunk,
    # and padding rows, are routed to a throwaway row.
    garbage_dst = out_hbm_ref.shape[0] - 1
    dma_src_rows = []
    dma_dst_rows = []
    for s in range(cfg.num_row_subchunks):
        sub_src = []
        sub_dst = []
        for i in range(num_simd_lanes):
            global_idx = s * num_simd_lanes + i
            merge_target = src_row_idx_in_vmem[global_idx]
            is_final_write = jnp.logical_and(
                row_valid_vec[global_idx],
                merge_target < (s + 1) * num_simd_lanes,
            )
            sub_src.append(
                jnp.where(is_final_write, merge_target % num_simd_lanes, 0))
            sub_dst.append(
                jnp.where(is_final_write, dst_indices_list[s][i], garbage_dst))
        dma_src_rows.append(sub_src)
        dma_dst_rows.append(sub_dst)

    for s in range(cfg.num_row_subchunks):
        sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
        scratch_ref.dma_src_row_vmem[sub] = _pack_scalars_to_vector(
            dma_src_rows[s])
        scratch_ref.dma_dst_row_vmem[sub] = _pack_scalars_to_vector(
            dma_dst_rows[s])

    col_pipeline = pltpu.emit_pipeline(
        functools.partial(
            _col_kernel,
            scratch_ref=scratch_ref,
            out_hbm_ref=out_hbm_ref,
            col_start=col_start,
            cfg=cfg,
        ),
        grid=(cfg.num_row_subchunks, cfg.num_col_chunks),
        in_specs=_col_gather_spec(scratch_ref.src_indices_vmem, col_start,
                                  cfg),
    )

    col_pipeline(in_32b_hbm_ref)
    scratch_ref.prev_dst_row_smem[0] = dst_indices_list[-1][num_simd_lanes - 1]


def _col_kernel(
    gather_ref: jax.Ref,
    *,
    scratch_ref: memory_ref.ScratchRef,
    out_hbm_ref: jax.Ref,
    col_start: jax.Array,
    cfg: config.Config,
):
    s = pl.program_id(0)
    c = pl.program_id(1)
    col_hbm_start = col_start + c * cfg.col_chunk_size
    send_sem = scratch_ref.sem.at[1]

    num_simd_lanes = cfg.sc_info.num_lanes

    row_slice = pl.ds(s * num_simd_lanes, num_simd_lanes)
    tw_slice = scratch_ref.tw_f32_vmem[row_slice]
    dst_slice = scratch_ref.dst_indices_vmem[row_slice]
    src_idx_slice = scratch_ref.src_indices_vmem[row_slice]
    prev_dst_vals_vec = scratch_ref.prev_dst_val_vmem[row_slice]

    @plsc.parallel_loop(0, cfg.col_chunk_size, step=num_simd_lanes)
    def col_loop(col_compute_offset: jax.Array):
        col_slice = pl.ds(col_compute_offset, num_simd_lanes)
        # Running sum, seeded by the carry from the previous sub-chunk.
        previous_accumulated_data = scratch_ref.prev_iter_last_row_vmem[
            c, col_slice]

        for row_src in range(num_simd_lanes):
            val_u32 = gather_ref[row_src, col_slice]
            if cfg.in_dtype == jnp.bfloat16:
                # The two bfloat16 rows packed in one uint32 word sit in the low
                # (even row) or high (odd row) 16 bits. Shift the wanted half
                # into the float32 sign/exponent position and clear the rest.
                is_even_row = jnp.bitwise_and(src_idx_slice[row_src], 1) == 0
                shift = jnp.where(is_even_row, 16, 0)
                # This is not really a problem in python, but using (1 << bits) - 1 to
                # create bitmask is not a safe practice as it can trigger overflow
                # depending on programming languages and data types.
                lower_mask = jnp.uint32(jnp.iinfo(jnp.uint16).max)
                upper_mask = jnp.left_shift(lower_mask, 16)
                shifted = jnp.bitwise_and(jnp.left_shift(val_u32, shift),
                                          upper_mask)
                data_f32 = plsc.bitcast(shifted, jnp.float32)
            else:
                data_f32 = plsc.bitcast(val_u32, jnp.float32)
            data_f32 *= tw_slice[row_src]

            # Reduction: accumulate while the destination group is unchanged,
            # restart otherwise. Sorting guarantees rows of one group are
            # contiguous.
            dst_row_hbm = dst_slice[row_src]
            if row_src == 0:
                prev_dst = prev_dst_vals_vec[0]
            else:
                prev_dst = dst_slice[row_src - 1]
            accumulated_data = jnp.where(
                dst_row_hbm == prev_dst,
                previous_accumulated_data + data_f32,
                data_f32,
            )
            previous_accumulated_data = accumulated_data

            # The output buffer stays float32: a bfloat16 output would be
            # (16, 128)-tiled and the per-row scatter below writes a single
            # row at an arbitrary, non-tile-aligned destination, which is only
            # legal for 32-bit elements. The cast happens in the wrapper.
            scratch_ref.out_vmem[row_src, col_slice] = accumulated_data
            if row_src == num_simd_lanes - 1:
                scratch_ref.prev_iter_last_row_vmem[
                    c, col_slice] = accumulated_data

    # Scatter every source row's reduced value to its output row. Rows
    # that share a group write the same value (idempotent); rows routed to
    # the garbage destination are harmless.
    dma_src_row_slice = scratch_ref.dma_src_row_vmem[row_slice]
    dma_dst_row_slice = scratch_ref.dma_dst_row_vmem[row_slice]
    copies = []
    for i in range(num_simd_lanes):
        copy = pltpu.make_async_copy(
            scratch_ref.out_vmem.at[dma_src_row_slice[i],
                                    pl.ds(0, cfg.col_chunk_size)],
            out_hbm_ref.at[dma_dst_row_slice[i],
                           pl.ds(col_hbm_start, cfg.col_chunk_size)],
            send_sem,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
