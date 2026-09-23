import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import config, memory_ref


def call_kernel_pipeline(
    row_partition_id: jax.Array,
    refs: memory_ref.KernelRefs,
    col_start: jax.Array,
    cfg: config.Config,
):
    num_rows_per_row_partition = refs.scratch.num_rows_per_row_partition_vmem[...]
    num_rows_current_row_partition = jnp.array(0, jnp.int32)
    for i in range(cfg.num_row_partitions):
        num_rows_current_row_partition = jnp.where(
            row_partition_id == i,
            num_rows_per_row_partition[i],
            num_rows_current_row_partition,
        )
    num_row_blocks = pl.cdiv(num_rows_current_row_partition, cfg.row_chunk_size)

    # Sentinel for the cross-block reduction carry (no previous group). The carry
    # is kernel scratch, so it also persists across window boundaries.
    refs.scratch.prev_dst_row_smem[0] = -1

    # The resident sort scratch holds one window of row blocks, so the row blocks
    # are run one window at a time. The window count is a runtime value (it follows
    # the row partition's row count), hence a pl.loop over the window index.
    num_windows = pl.cdiv(num_row_blocks, cfg.max_window)
    pl.loop(0, num_windows)(
        functools.partial(
            _window_kernel,
            refs=refs,
            row_partition_id=row_partition_id,
            num_row_blocks=num_row_blocks,
            num_rows_current_row_partition=num_rows_current_row_partition,
            col_start=col_start,
            cfg=cfg,
        )
    )


def _window_kernel(
    window_id: jax.Array,
    *,
    refs: memory_ref.KernelRefs,
    row_partition_id: jax.Array,
    num_row_blocks: jax.Array,
    num_rows_current_row_partition: jax.Array,
    col_start: jax.Array,
    cfg: config.Config,
):
    """Stages one window of the sort permutation, then runs its row blocks."""
    num_simd_lanes = cfg.sc_info.num_lanes
    sorted_by_validity = refs.index.sorted_by_validity
    recv_sem = refs.scratch.sem.at[0]
    window_words = cfg.window_size
    window_start = (
        row_partition_id * cfg.row_partition_size_padded + window_id * window_words
    )
    window_block_base = window_id * cfg.max_window

    # Streaming one window at a time bounds the resident scratch (it no longer
    # grows with the input size). The whole window is staged before the pipeline,
    # so the pl.Indirect source has no hazard.
    sorted_dma = pltpu.make_async_copy(
        sorted_by_validity.at[pl.ds(window_start, window_words)],
        refs.scratch.sorted_by_validity_vmem,
        recv_sem,
    )
    sorted_dma.start()
    sorted_dma.wait()

    # Prefetch the next window's first source row so the last block of this window
    # can detect a group continuing across the window boundary. Clamped, so the
    # last window fetches a harmless past-the-end row instead.
    last_start = sorted_by_validity.shape[0] - num_simd_lanes
    next_window_start = jnp.minimum(window_start + window_words, last_start)
    next_window_rows = sorted_by_validity.at[pl.ds(next_window_start, num_simd_lanes)]
    next_window_dma = pltpu.make_async_copy(
        next_window_rows,
        refs.scratch.next_window_first_row_vmem,
        recv_sem,
    )
    next_window_dma.start()
    next_window_dma.wait()

    blocks_in_window = jnp.minimum(cfg.max_window, num_row_blocks - window_block_base)

    # The index maps read the resident sort window staged above.
    row_gather_specs = _row_gather_specs(refs.scratch.sorted_by_validity_vmem, cfg)
    row_pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            _row_kernel,
            cfg=cfg,
            refs=refs,
            num_rows_current_row_partition=num_rows_current_row_partition,
            col_start=col_start,
            window_block_base=window_block_base,
            blocks_in_window=blocks_in_window,
        ),
        grid=(blocks_in_window,),
        in_specs=(row_gather_specs, row_gather_specs),
    )
    row_pipeline_fn(
        ((refs.index.indices,) * cfg.num_row_subchunks),
        ((refs.data.topk_weights,) * cfg.num_row_subchunks),
    )


def _pack_scalars_to_vector(scalar_list: list[jax.Array]) -> jax.Array:
    """Pack list of scalar values into a single VMEM lane."""
    num_lanes = len(scalar_list)
    idx_vec = jax.lax.broadcasted_iota(jnp.int32, (num_lanes,), 0)
    vec = jnp.zeros((num_lanes,), jnp.int32)
    for i in range(num_lanes):
        vec = jnp.where(idx_vec == i, scalar_list[i], vec)
    return vec


def _row_gather_specs(
    sorted_by_validity_vmem: jax.Ref, cfg: config.Config
) -> tuple[pl.BlockSpec, ...]:
    """Indirect BlockSpec gathering rows of a 1-D input."""
    num_simd_lanes = cfg.sc_info.num_lanes

    def row_index_map(r: int | jax.Array, *, offset: int) -> jax.Array:
        start = r * cfg.row_chunk_size + offset * num_simd_lanes
        return sorted_by_validity_vmem[pl.ds(start, num_simd_lanes)]

    return tuple(
        [
            pl.BlockSpec(
                (pl.Indirect(num_simd_lanes),),
                functools.partial(row_index_map, offset=offset),
            )
            for offset in range(cfg.num_row_subchunks)
        ]
    )


def _col_gather_spec(
    src_indices_vmem: jax.Ref,
    col_start: jax.Array,
    cfg: config.Config,
) -> pl.BlockSpec:
    """Indirect BlockSpec gathering columns of a 2-D input."""
    num_simd_lanes = cfg.sc_info.num_lanes

    def col_index_map(
        s: int | jax.Array, c: int | jax.Array
    ) -> tuple[jax.Array, jax.Array]:
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
    refs: memory_ref.KernelRefs,
    num_rows_current_row_partition: jax.Array,
    col_start: jax.Array,
    window_block_base: jax.Array,
    blocks_in_window: jax.Array,
    cfg: config.Config,
):
    # The SparseCore indirect DMA requires 32-bit elements, so x is gathered
    # through a uint32 reinterpretation. bfloat16 packs two source rows per
    # uint32 row (row index >> 1); float32 is 1:1 (row index unchanged).
    in_32b_hbm_ref = refs.data.source.bitcast(jnp.uint32)
    num_simd_lanes = cfg.sc_info.num_lanes
    window_words = cfg.window_size
    row_block_id = pl.program_id(0)
    # Absolute row-block index within the partition (for the validity mask); the
    # resident sort window is indexed window-relative.
    global_block_id = window_block_base + row_block_id

    # Destination output row of each source row in this block.
    dst_indices_list = []
    for s in range(cfg.num_row_subchunks):
        start = row_block_id * cfg.row_chunk_size + s * num_simd_lanes
        gather_dst = refs.scratch.sorted_by_validity_vmem[pl.ds(start, num_simd_lanes)]
        dst_indices_list.append(gather_dst // cfg.reduce_group_size)

    # Stage the gathered indices/weights and the destinations in VMEM.
    for s in range(cfg.num_row_subchunks):
        sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
        refs.scratch.dst_indices_vmem[sub] = dst_indices_list[s]
        # Store list of VMEMs into a continuous data to enable dynamic indexing.
        # TODO(kyuyeunk): Convert DMA destination into a continuous data to avoid
        # the performance and memory overhead of ldst.
        refs.scratch.src_indices_vmem[sub] = src_indices_refs[s][...]
        refs.scratch.tw_f32_vmem[sub] = topk_weights_refs[s][...]

    # For each sub-chunk, the destination of the row just before it -- the
    # seed for the segmented reduction's "same group as previous row" test.
    for s in range(cfg.num_row_subchunks):
        if s == 0:
            prev_dst = refs.scratch.prev_dst_row_smem[0]
        else:
            prev_dst = dst_indices_list[s - 1][num_simd_lanes - 1]
        refs.scratch.prev_dst_val_vmem[pl.ds(s * num_simd_lanes, num_simd_lanes)] = (
            jnp.broadcast_to(prev_dst, (num_simd_lanes,))
        )

    # For each source row, find the VMEM row that will hold its group's fully
    # reduced value -- the last row of the group within this block. Scanning
    # backwards, a row inherits its successor's merge target when they share
    # a destination, otherwise it is its own target.
    rev_src_row_idx_in_vmem = []
    rev_is_row_valid = []
    for row_vmem_idx in reversed(range(cfg.row_chunk_size)):
        if row_vmem_idx == cfg.row_chunk_size - 1:
            next_src_row_idx = row_vmem_idx
        else:
            quot, rem = divmod(row_vmem_idx, num_simd_lanes)
            quot_next, rem_next = divmod(row_vmem_idx + 1, num_simd_lanes)
            same_group_as_next = jnp.logical_and(
                rev_is_row_valid[-1],
                dst_indices_list[quot][rem] == dst_indices_list[quot_next][rem_next],
            )
            next_src_row_idx = jnp.where(
                same_group_as_next, rev_src_row_idx_in_vmem[-1], row_vmem_idx
            )
        global_row_idx = global_block_id * cfg.row_chunk_size + row_vmem_idx
        rev_is_row_valid.append(global_row_idx < num_rows_current_row_partition)
        rev_src_row_idx_in_vmem.append(next_src_row_idx)
    src_row_idx_in_vmem = rev_src_row_idx_in_vmem[::-1]
    is_row_valid = rev_is_row_valid[::-1]

    # A reduce group whose last in-block row is the block's final row may continue
    # into the next block, which writes the group's full reduced value. Suppress
    # this block's partial write for such a group so each output row has a single
    # writer -- neither the pipelined row-block emit_pipeline nor the outer window
    # loop orders the two scatters, so a double write races on the output row.
    is_last_block_in_window = (row_block_id + 1) == blocks_in_window
    next_block_first_row = (global_block_id + 1) * cfg.row_chunk_size
    # The next block's first source row sits in the resident sort window, except on
    # the window's last block, where it is the prefetched first row of the next
    # window. Both arms of the select are evaluated, so the resident offset is
    # clamped to stay in bounds on that last block (where its value is unused).
    next_block_first_row_in_window = jnp.minimum(
        (row_block_id + 1) * cfg.row_chunk_size, window_words - num_simd_lanes
    )
    next_block_first_idx = jnp.where(
        is_last_block_in_window,
        refs.scratch.next_window_first_row_vmem[...][0],
        refs.scratch.sorted_by_validity_vmem[
            pl.ds(next_block_first_row_in_window, num_simd_lanes)
        ][0],
    )
    group_continues = jnp.logical_and(
        next_block_first_row < num_rows_current_row_partition,
        (next_block_first_idx // cfg.reduce_group_size)
        == dst_indices_list[-1][num_simd_lanes - 1],
    )

    # Per source row, the (VMEM source row, HBM destination row) of its
    # scatter. Rows whose group is not yet fully reduced in this sub-chunk,
    # and padding rows, are routed to a throwaway row.
    garbage_dst = refs.data.out.shape[0] - 1
    dma_src_rows = []
    dma_dst_rows = []
    for s in range(cfg.num_row_subchunks):
        sub_src = []
        sub_dst = []
        for i in range(num_simd_lanes):
            global_idx = s * num_simd_lanes + i
            merge_target = src_row_idx_in_vmem[global_idx]
            is_final_write = jnp.logical_and(
                is_row_valid[global_idx],
                merge_target < (s + 1) * num_simd_lanes,
            )
            # Only the last sub-chunk's group can reach the block's final row; earlier
            # sub-chunks already route such a group to garbage.
            if s == cfg.num_row_subchunks - 1:
                merges_at_block_end = merge_target == cfg.row_chunk_size - 1
                spans_next_block = jnp.logical_and(merges_at_block_end, group_continues)
                is_final_write = jnp.logical_and(
                    is_final_write, jnp.logical_not(spans_next_block)
                )
            sub_src.append(jnp.where(is_final_write, merge_target % num_simd_lanes, 0))
            sub_dst.append(
                jnp.where(is_final_write, dst_indices_list[s][i], garbage_dst)
            )
        dma_src_rows.append(sub_src)
        dma_dst_rows.append(sub_dst)

    for s in range(cfg.num_row_subchunks):
        sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
        refs.scratch.dma_src_row_vmem[sub] = _pack_scalars_to_vector(dma_src_rows[s])
        refs.scratch.dma_dst_row_vmem[sub] = _pack_scalars_to_vector(dma_dst_rows[s])

    col_pipeline = pltpu.emit_pipeline(
        functools.partial(
            _col_kernel,
            refs=refs,
            col_start=col_start,
            cfg=cfg,
        ),
        grid=(cfg.num_row_subchunks, cfg.num_col_chunks),
        in_specs=_col_gather_spec(refs.scratch.src_indices_vmem, col_start, cfg),
    )
    col_pipeline(in_32b_hbm_ref)
    refs.scratch.prev_dst_row_smem[0] = dst_indices_list[-1][num_simd_lanes - 1]


def _col_kernel(
    gather_ref: jax.Ref,
    *,
    refs: memory_ref.KernelRefs,
    col_start: jax.Array,
    cfg: config.Config,
):
    s = pl.program_id(0)
    c = pl.program_id(1)
    col_hbm_start = col_start + c * cfg.col_chunk_size
    send_sem = refs.scratch.sem.at[1]
    num_simd_lanes = cfg.sc_info.num_lanes

    row_slice = pl.ds(s * num_simd_lanes, num_simd_lanes)
    tw_slice = refs.scratch.tw_f32_vmem[row_slice]
    dst_slice = refs.scratch.dst_indices_vmem[row_slice]
    src_idx_slice = refs.scratch.src_indices_vmem[row_slice]
    prev_dst_vals_vec = refs.scratch.prev_dst_val_vmem[row_slice]

    @plsc.parallel_loop(0, cfg.col_chunk_size, step=num_simd_lanes)
    def col_loop(col_compute_offset: jax.Array):
        col_slice = pl.ds(col_compute_offset, num_simd_lanes)
        # Running sum, seeded by the carry from the previous sub-chunk.
        previous_accumulated_data = refs.scratch.prev_iter_last_row_vmem[c, col_slice]
        for row_src in range(num_simd_lanes):
            val_u32 = gather_ref[row_src, col_slice]
            if cfg.in_dtype == jnp.bfloat16:
                # The two bfloat16 rows packed in one uint32 word sit in the low
                # (even row) or high (odd row) 16 bits. Shift the wanted half
                # into the float32 sign/exponent position and clear the rest.
                is_even_row = jnp.bitwise_and(src_idx_slice[row_src], 1) == 0
                shift = jnp.where(is_even_row, 16, 0)
                lower_mask = jnp.uint32(jnp.iinfo(jnp.uint16).max)
                upper_mask = jnp.left_shift(lower_mask, 16)
                shifted = jnp.bitwise_and(jnp.left_shift(val_u32, shift), upper_mask)
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
            refs.scratch.out_vmem[row_src, col_slice] = accumulated_data
            if row_src == num_simd_lanes - 1:
                refs.scratch.prev_iter_last_row_vmem[c, col_slice] = accumulated_data

    # Scatter every source row's reduced value to its output row. Rows
    # that share a group write the same value (idempotent); rows routed to
    # the garbage destination are harmless.
    dma_src_row_slice = refs.scratch.dma_src_row_vmem[row_slice]
    dma_dst_row_slice = refs.scratch.dma_dst_row_vmem[row_slice]
    copies = []
    for i in range(num_simd_lanes):
        copy = pltpu.make_async_copy(
            refs.scratch.out_vmem.at[
                dma_src_row_slice[i], pl.ds(0, cfg.col_chunk_size)
            ],
            refs.data.out.at[
                dma_dst_row_slice[i], pl.ds(col_hbm_start, cfg.col_chunk_size)
            ],
            send_sem,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
