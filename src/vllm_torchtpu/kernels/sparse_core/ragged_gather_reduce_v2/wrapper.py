# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
    config,
    kernel,
    memory_ref,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
    ragged_gather_reduce as ragged_gather_reduce_v3,
    token_block_alignment,
)


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

    # Pack the 1-bit validity flag into bit 30 and the strictly increasing
    # global row index into bits 0..29 so a single-key unstable sort produces
    # the exact stable partition + global offset in one pass.
    global_iota = jnp.arange(cfg.padded_input_size, dtype=jnp.int32).reshape(
        cfg.num_row_partitions, cfg.row_partition_size
    )
    packed = jnp.where(valid_rows_mask_2d, global_iota, global_iota | (1 << 30))
    sorted_by_validity = jnp.bitwise_and(
        jax.lax.sort(packed, dimension=-1, is_stable=False),
        (1 << 30) - 1,
    )

    padding = cfg.row_partition_size_padded - cfg.row_partition_size
    sorted_by_validity = jnp.pad(sorted_by_validity, ((0, 0), (0, padding)))
    sorted_by_validity = sorted_by_validity.reshape(-1)

    num_src_rows_per_row_partition = jnp.pad(
        jnp.sum(valid_rows_mask_2d, axis=-1).astype(jnp.int32),
        (0, max(0, num_simd_lanes - cfg.num_row_partitions)),
    )
    if cfg.reduce_group_size == 1:
        mask = valid_rows_mask
    else:
        # Avoid reshaping 1D `pred[N]{0:T(1024)}` to 2D `pred[N//G, G]{1,0:T(8,128)}`
        # (95.3% lane padding waste for G=6). Reshape to 128-lane-aligned 3D
        # `(-1, G, 128)` (zero-copy bitcast) and contract on the MXU.
        g = cfg.reduce_group_size
        num_groups = cfg.padded_input_size // g
        block_elems = g * 128
        rem = cfg.padded_input_size % block_elems
        v_pad = (
            jnp.pad(valid_rows_mask, (0, block_elems - rem))
            if rem
            else valid_rows_mask
        )
        v_3d = v_pad.reshape(-1, g, 128).astype(jnp.bfloat16)
        pat = (
            (jnp.arange(block_elems)[:, None] // g == jnp.arange(128)[None, :])
            .reshape(g, 128, 128)
            .astype(jnp.bfloat16)
        )
        mask = (
            jnp.einsum(
                "brc,rcg->bg", v_3d, pat, preferred_element_type=jnp.float32
            )
            > 0
        ).reshape(-1)[:num_groups]
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
    core_id = jax.lax.axis_index((cfg.core_axis_name, cfg.subcore_axis_name))
    row_partition_id = core_id // cfg.num_column_partitions
    col_partition_id = core_id % cfg.num_column_partitions

    col_start = col_partition_id * cfg.col_size

    refs = memory_ref.KernelRefs.create(
        scalar_ref=scalar_ref,
        in_hbm_ref=in_hbm_ref,
        topk_weights_hbm_ref=topk_weights_hbm_ref,
        out_hbm_ref=out_hbm_ref,
        scratch_ref=scratch_ref,
    )

    # Step 2: Stage this partition's row count (the sort permutation is streamed
    # one window at a time in call_kernel_pipeline, bounding the resident scratch).
    recv_sem = refs.scratch.sem.at[0]
    num_rows_dma = pltpu.make_async_copy(
        refs.index.num_src_rows_per_row_partition,
        refs.scratch.num_rows_per_row_partition_vmem,
        recv_sem,
    )
    num_rows_dma.start()
    num_rows_dma.wait()

    kernel.call_kernel_pipeline(
        row_partition_id=row_partition_id,
        refs=refs,
        col_start=col_start,
        cfg=cfg,
    )


@jax.jit(static_argnames=("reduce_group_size",))
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
        source_rows=x.shape[0],
        reduce_group_size=reduce_group_size,
        in_dtype=x.dtype,
        core_axis_name="core",
        subcore_axis_name="subcore",
        tpu_info=pltpu.get_tpu_info(),
    )

    # Step 2: Fallback to compiler version if needed.
    if cfg.should_fallback:
        return _fallback_implementation(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )

    if (
        x.dtype == jnp.bfloat16
        and reduce_group_size >= 2
        and x.shape[0] == cfg.input_size
        and x.shape[0] <= (1 << 20)
        and cfg.sc_info.num_lanes == 16
        and cfg.hidden_size % cfg.tpu_info.num_lanes == 0
    ):
        alignment = token_block_alignment(cfg.hidden_size, cfg.tpu_info)
        if cfg.output_size >= alignment and cfg.output_size % alignment == 0:
            return ragged_gather_reduce_v3(
                x, indices, topk_weights, valid_rows_mask, reduce_group_size
            )

    # Step 3: Pre-process inputs (weights, padding, sort by validity).
    # Simplify topk gather by using fp32 and ensure data is always word aligned.
    topk_weights_f32 = topk_weights.astype(jnp.float32)

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
    out = core_map_helper.kernel(
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
        scratch_types=(memory_ref.ScratchRef.create_scratch_types(cfg),),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v2",
    )(scalar, x, topk_weights_f32)

    # Step 5: Post-process the output (drop padding, zero empty groups, cast).
    out = out[: cfg.output_size, : cfg.hidden_size].astype(x.dtype)
    # TODO(kyuyeunk): Use zero initialization instead of masking.
    return jnp.where(mask[: cfg.output_size, None], out, 0)
