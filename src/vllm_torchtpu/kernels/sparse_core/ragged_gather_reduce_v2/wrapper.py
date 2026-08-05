import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
    config, kernel, memory_ref)


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


def _preprocess_scalar_data(
    indices: jax.Array,
    valid_rows_mask: jax.Array,
    cfg: config.Config,
) -> tuple[memory_ref.IndexRef, jax.Array]:
    """Sorts valid source rows to the front of each row partition.

  Args:
    indices: Indices for gather.
    valid_rows_mask: Mask indicating valid rows.
    cfg: Ragged gather reduce config.

  Returns:
    sorted_by_validity: original row index of each slot after the stable
      sort, flattened across partitions and padded to ``row_chunk_size``.
    num_src_rows_per_row_partition: valid row count per partition, padded to
      ``num_simd_lanes`` so the kernel can load it as a single vector.
    mask: per output group, whether the group has any valid source row.
  """
    num_simd_lanes = cfg.sc_info.num_lanes
    valid_rows_mask_2d = valid_rows_mask.reshape(cfg.num_row_partitions, -1)

    # Stable sort of a boolean key is a stable partition: valid rows keep their
    # relative order and move ahead of the invalid ones.
    sorted_by_validity = jnp.argsort(~valid_rows_mask_2d,
                                     descending=False,
                                     stable=True,
                                     axis=-1)
    sorted_by_validity += (jnp.arange(cfg.num_row_partitions)[:, None] *
                           cfg.row_partition_size)

    padding = cfg.row_partition_size_padded - cfg.row_partition_size
    sorted_by_validity = jnp.pad(sorted_by_validity, ((0, 0), (0, padding)))
    sorted_by_validity = sorted_by_validity.reshape(-1)

    num_src_rows_per_row_partition = jnp.pad(
        jnp.sum(valid_rows_mask_2d, axis=-1).astype(jnp.int32),
        (0, max(0, num_simd_lanes - cfg.num_row_partitions)),
    )
    mask = jnp.any(valid_rows_mask.reshape(-1, cfg.reduce_group_size), axis=-1)
    scalar = memory_ref.IndexRef(
        indices=indices,
        sorted_by_validity=sorted_by_validity.astype(jnp.int32),
        num_src_rows_per_row_partition=num_src_rows_per_row_partition,
    )
    return scalar, mask


def main_kernel(
    scalar_ref: memory_ref.IndexRef,
    in_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    scratch_ref: memory_ref.ScratchRef,
    *,
    cfg: config.Config,
):
    # Step 1: Resolve this core's row/column partition and its column slice.
    num_simd_lanes = cfg.sc_info.num_lanes

    core_id = jax.lax.axis_index((cfg.core_axis_name, cfg.subcore_axis_name))
    row_partition_id = core_id // cfg.num_column_partitions
    col_partition_id = core_id % cfg.num_column_partitions

    row_start_padded = row_partition_id * cfg.row_partition_size_padded
    col_start = col_partition_id * cfg.col_size

    # Step 2: Stage this partition's row count and sort permutation into VMEM.
    recv_sem = scratch_ref.sem.at[0]
    num_rows_dma = pltpu.make_async_copy(
        scalar_ref.num_src_rows_per_row_partition.at[pl.ds(0, num_simd_lanes)],
        scratch_ref.num_rows_per_row_partition_vmem,
        recv_sem,
    )
    sorted_dma = pltpu.make_async_copy(
        scalar_ref.sorted_by_validity.at[pl.ds(row_start_padded,
                                               cfg.row_partition_size_padded)],
        scratch_ref.sorted_by_validity_vmem,
        recv_sem,
    )
    num_rows_dma.start()
    sorted_dma.start()
    num_rows_dma.wait()
    sorted_dma.wait()

    kernel.call_kernel_pipeline(
        row_partition_id=row_partition_id,
        scalar_ref=scalar_ref,
        scratch_ref=scratch_ref,
        in_hbm_ref=in_hbm_ref,
        out_hbm_ref=out_hbm_ref,
        topk_weights_hbm_ref=topk_weights_hbm_ref,
        col_start=col_start,
        cfg=cfg,
    )


@jax.jit(static_argnames=("reduce_group_size", ))
def ragged_gather_reduce_v2(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Gathers ``x`` by ``indices``, weights and masks, then reduces by group.

  Args:
    x: 2-D input features, ``(num_rows, hidden_size)``.
    indices: 1-D gather indices, ``(input_size,)``.
    topk_weights: 1-D per-row weights, ``(input_size,)``.
    valid_rows_mask: 1-D bool mask of valid gathered rows, ``(input_size,)``.
    reduce_group_size: number of consecutive rows summed into one output row.

  Returns:
    Reduced output, ``(input_size // reduce_group_size, hidden_size)``.
  """

    # Step 1: Create config object.
    cfg = config.Config(
        input_size=indices.size,
        hidden_size=x.shape[-1],
        reduce_group_size=reduce_group_size,
        topk_dtype=topk_weights.dtype,
        in_dtype=x.dtype,
        core_axis_name="core",
        subcore_axis_name="subcore",
        tpu_info=pltpu.get_tpu_info(),
    )

    # Step 2: Fallback to compiler version if needed.
    if cfg.should_fallback:
        return _fallback_implementation(x, indices, topk_weights,
                                        valid_rows_mask, reduce_group_size)

    # Step 3: Pre-process inputs (weights, padding, sort by validity).
    # The kernel gathers x through a uint32 reinterpretation; carry the weights
    # the same way so they can be bitcast back to float32 on SparseCore.
    if topk_weights.dtype == jnp.bfloat16:
        topk_weights_u32 = jax.lax.bitcast_convert_type(
            topk_weights, jnp.uint16).astype(jnp.uint32)
    else:
        topk_weights_u32 = jax.lax.bitcast_convert_type(
            topk_weights, jnp.uint32)

    # Pad the input so each row partition holds a whole number of reduce
    # groups; no group is then split across two physical cores.
    valid_rows_mask = jnp.pad(
        valid_rows_mask,
        (0, cfg.padded_input_size - cfg.input_size),
        constant_values=False,
    )

    scalar, mask = _preprocess_scalar_data(indices, valid_rows_mask, cfg)

    # Step 4: Launch the SparseCore kernel.
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=cfg.sc_info.num_cores,
        num_subcores=cfg.sc_info.num_subcores,
        core_axis_name=cfg.core_axis_name,
        subcore_axis_name=cfg.subcore_axis_name,
    )

    # The output gets one extra row: the kernel's garbage scatter destination.
    out = pl.kernel(
        functools.partial(main_kernel, cfg=cfg),
        out_type=jax.ShapeDtypeStruct(
            (
                cfg.padded_input_size // reduce_group_size + 1,
                cfg.aligned_hidden_size,
            ),
            jnp.float32,
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=(memory_ref.ScratchRef.create_scratch_types(cfg), ),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v2",
    )(scalar, x, topk_weights_u32)

    # Step 5: Post-process the output (drop padding, zero empty groups, cast).
    out = out[:cfg.output_size, :cfg.hidden_size]
    # TODO(kyuyeunk): Use zero initialization instead of masking.
    out = jnp.where(mask[:cfg.output_size, None], out, jnp.zeros_like(out))
    return out.astype(x.dtype)
