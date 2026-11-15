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

import dataclasses
import functools
import math
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import (
    compute_conv1d,
    compute_gdn,
    compute_kda,
    config,
    memory_ref,
    metadata,
    tiling,
    vmem_ldst,
    window_tile,
)
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)
# Trace-time record of which opt-in raw/zero-out paths were taken (log once each).
_RAW_PATH_LOGGED: set = set()


def _load_raw_rows(
    block_ref: jax.Ref,  # [rows, width] raw row block (RowBlockInBufferedRef)
    stage_ref: jax.Ref,  # [rows + chunk, 1, padded_width] float32 scratch
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    cfg: config.GDNConfig,
    align: int,
) -> jax.Array:
    """Casts a raw row block to FP32 and gathers each sequence's rows.

    Returns [seq_tile_size, chunk_size, 1, padded_width] float32, the value the
    prepared [tokens, 1, padded_width] FP32 input's slot would hold: the
    same cast (exact), and lanes past `width` zero like the wrapper's pad.
    Rows past a sequence's real size hold neighbouring rows instead of
    stale data; the kernel masks them either way.
    """
    rows, width = block_ref.shape
    padded_width = stage_ref.shape[-1]
    num_lanes = pltpu.get_tpu_info().num_lanes
    start = memory_ref.raw_block_start(
        metadata_ref, p_id, rows, align, cfg.batch_size
    )
    if width == padded_width:
        for col in range(0, width, num_lanes):
            stage_ref[pl.ds(0, rows), 0, col : col + num_lanes] = block_ref[
                :, col : col + num_lanes
            ].astype(jnp.float32)
    else:
        block = block_ref[...].astype(jnp.float32)
        stage_ref[pl.ds(0, rows), 0, :] = jnp.pad(
            block, ((0, 0), (0, padded_width - width))
        )
    parts = []
    for idx in range(cfg.seq_tile_size):
        offset = jnp.clip(metadata_ref.p_id_to_r_base[p_id, idx] - start, 0, rows)
        parts.append(stage_ref[pl.ds(offset, cfg.chunk_size)])
    return jnp.stack(parts, axis=0)


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size]
    b_slot_ref: jax.Ref,  # [seq, chunk, 1, num_v_heads]
    a_slot_ref: jax.Ref,  # [seq, chunk, 1, num_v_heads]
    conv_state_in_slot_ref: jax.Ref,  # [seq, 1, prev_kernel_size, 1, dim_size]
    recurrent_in_slot_ref: jax.Ref,  # [seq, 1, num_v_heads, kq_head, v_head]
    # Outputs.
    out_slot_ref: jax.Ref,  # [seq, chunk, num_v_heads, v_head]
    conv_state_out_slot_ref: jax.Ref,  # [seq, window, prev_kernel_size, 1, dim]
    recurrent_out_slot_ref: jax.Ref,  # [seq, window, num_v_heads, kq, v]
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    raw_scratch_ref: dict[str, jax.Ref] | None = None,
    window_scratch_refs: tuple[jax.Ref, ...] | None = None,
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array | None = None,
):
    """Orchestrates computation of Conv1D and GDN for a single tile.

    This kernel acts as a facade adhering to strict separation of concerns. It
    operates VMEM reference without knowledge on DMA logic. Furthermore, the
    kernel invokes vmem_ldst to pre-processes data needed for compute and
    invokes compute_conv1d and compute_gdn for actual compute.

    Args:
        qkv_slot_ref: qkv VMEM ref that stores data loaded from HBM.
        b_slot_ref: b VMEM ref that stores data loaded from HBM.
        a_slot_ref: a VMEM ref that stores data loaded from HBM.
        conv_state_in_slot_ref: Convolution state VMEM ref loaded from HBM.
        recurrent_in_slot_ref: Recurrent state VMEM ref loaded from HBM.
        out_slot_ref: Output VMEM ref that will be used for VMEM to HBM write.
        conv_state_out_slot_ref: Convolution state VMEM ref written to HBM.
        recurrent_out_slot_ref: Recurrent state VMEM ref written to HBM.
        metadata_ref: Metadata reference containing grid and sequence mappings.
        weights_ref: Weight references for Conv1D and GDN in VMEM.
        carry_conv_scratch_ref: Optional VMEM scratch reference for inter-tile
            convolution carry.
        carry_recurrent_scratch_ref: Optional VMEM scratch reference for inter-tile
            recurrent state carry.
        window_scratch_refs: window_tile's VMEM scratches (head states,
            conv inputs, conv weights) when `cfg.per_head_windows`.
        cfg: GDN configuration object.
        p_id: Optional explicit tile index. Communication-fused outer kernels
            pass their global compute-tile index because their surrounding
            pipeline owns the Pallas program id.
    """

    if p_id is None:
        p_id = pl.program_id(0)

    if cfg.per_head_windows:
        # Verify-window tile, one position and one value head at a time
        # (VLLM_TPU_GDN_V3_WINDOW_VERIFY). Raw inputs are staged to the
        # FP32 tiles inner_kernel would read.
        assert window_scratch_refs is not None
        if cfg.raw_qkv:
            assert raw_scratch_ref is not None
            raw_scratch_ref["qkv_act"][...] = _load_raw_rows(
                qkv_slot_ref,
                raw_scratch_ref["qkv_stage"],
                metadata_ref,
                p_id,
                cfg,
                cfg.raw_qkv_align,
            )
            qkv_slot_ref = raw_scratch_ref["qkv_act"]
        if cfg.raw_ba:
            assert raw_scratch_ref is not None
            for name, raw_ref in (("b", b_slot_ref), ("a", a_slot_ref)):
                raw_scratch_ref[f"{name}_act"][...] = _load_raw_rows(
                    raw_ref,
                    raw_scratch_ref[f"{name}_stage"],
                    metadata_ref,
                    p_id,
                    cfg,
                    cfg.raw_ba_align,
                )
            b_slot_ref = raw_scratch_ref["b_act"]
            a_slot_ref = raw_scratch_ref["a_act"]
        window_tile.run(
            qkv_slot_ref,
            b_slot_ref,
            a_slot_ref,
            conv_state_in_slot_ref,
            recurrent_in_slot_ref,
            out_slot_ref,
            conv_state_out_slot_ref,
            recurrent_out_slot_ref,
            metadata_ref,
            weights_ref,
            *window_scratch_refs,
            cfg=cfg,
            p_id=p_id,
        )
        return

    # Prepare states.
    real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_in_slot_ref=conv_state_in_slot_ref,
        recurrent_in_slot_ref=recurrent_in_slot_ref,
        carry_conv_scratch_ref=carry_conv_scratch_ref,
        carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
        cfg=cfg,
    )

    # Step 1: Conv1D.
    # NOTE: Conv1D requires performing sliding window where inputs are slided
    # across rows. If typical 2D layout was used, multiple rows are stored in a
    # single register which necessitate costly shuffling for every sliding.
    # Therefore, it is extremely important to leverage compact layout that
    # ensures 1 register only stores data from 1 row.
    if cfg.raw_qkv:
        assert raw_scratch_ref is not None
        qkv_in_compact = _load_raw_rows(
            qkv_slot_ref,
            raw_scratch_ref["qkv_stage"],
            metadata_ref,
            p_id,
            cfg,
            cfg.raw_qkv_align,
        )
        # FP32 compact scratch the activation loaders stage the conv output
        # through (the raw block is neither FP32 nor compact).
        qkv_slot_ref = raw_scratch_ref["qkv_act"]
    else:
        qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
    if cfg.raw_ba:
        assert raw_scratch_ref is not None
        for name, raw_ref in (("b", b_slot_ref), ("a", a_slot_ref)):
            act_ref = raw_scratch_ref[f"{name}_act"]
            act_ref[...] = _load_raw_rows(
                raw_ref,
                raw_scratch_ref[f"{name}_stage"],
                metadata_ref,
                p_id,
                cfg,
                cfg.raw_ba_align,
            )
        b_slot_ref = raw_scratch_ref["b_act"]
        a_slot_ref = raw_scratch_ref["a_act"]
    qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

    # Prepare conv1d weights.
    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    conv_bias = None
    if weights_ref.conv.bias is not None:
        conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

    qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
        real_sizes=real_sizes,
        lhs=qkv_in_compact,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        cfg=cfg,
    )

    if cfg.state_plan is None:
        conv_state_out_slot_ref[...] = new_conv_state
    else:
        # External source tiles: encode each window position's checkpoint
        # through the region's typed view; copy_out pushes the raw bytes
        # of the valid checkpoints back to the source.
        for idx in range(cfg.seq_tile_size):
            for t in range(cfg.window_size):
                args = (
                    conv_state_out_slot_ref.at[idx, t],
                    cfg.state_plan.conv,
                    new_conv_state[idx, t],
                )
                if carry_conv_scratch_ref is None:
                    vmem_ldst.store_state_region(*args)
                else:
                    # copy_out only DMAs the tile that ends a sequence.
                    pl.when(metadata_ref.p_id_is_last_tile[p_id, idx])(
                        functools.partial(vmem_ldst.store_state_region, *args)
                    )
    if carry_conv_scratch_ref is not None:
        # The next tile resumes from the state after this tile's last token,
        # which is the final checkpoint.
        carry_conv_scratch_ref[...] = new_conv_state[:, -1]

    # Apply activation function.
    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    # Step 2: GDN.

    # Prepare gdn weights.
    if cfg.is_kda:
        # KDA reshapes these per (head, channel) rather than padding out to
        # a lane multiple over heads.
        a_log = weights_ref.gdn.a_log[...]
        dt_bias = weights_ref.gdn.dt_bias[...]
    else:
        padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
        a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
        dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

    # NOTE: Ideally, we want to move this branching logic into gdn.py. However,
    # load_activation_as_compact and load_activation_as_large leverages vmem ldst.
    # Passing refs into gdn.py breaks strict separation of concerns.
    if cfg.use_recurrent:
        recurrent_fn = (
            compute_kda.recurrent_kda if cfg.is_kda else compute_gdn.recurrent_gdn
        )
        q_compact, k_compact, v_compact, b_compact, a_compact = (
            vmem_ldst.load_activation_as_compact(
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfg,
            )
        )

        out, new_recurrent_state = recurrent_fn(
            q_compact=q_compact,
            k_compact=k_compact,
            v_compact=v_compact,
            b_compact=b_compact,
            a_compact=a_compact,
            state_prev=prev_recurrent,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            real_sizes=real_sizes,
        )

    else:
        chunked_fn = compute_kda.chunked_kda if cfg.is_kda else compute_gdn.chunked_gdn
        q_large, k_large, v_large, b_large, a_large = (
            vmem_ldst.load_activation_as_large(
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfg,
            )
        )

        out, new_recurrent_state = chunked_fn(
            q_large=q_large,
            k_large=k_large,
            v_large=v_large,
            b_large=b_large,
            a_large=a_large,
            state_prev=prev_recurrent,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            real_sizes=real_sizes,
        )

    # Store output and recurrent to vmem.
    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    if cfg.state_plan is None:
        recurrent_out_slot_ref[...] = new_recurrent_state.astype(
            recurrent_out_slot_ref.dtype
        )
    else:
        for idx in range(cfg.seq_tile_size):
            for t in range(cfg.window_size):
                args = (
                    recurrent_out_slot_ref.at[idx, t],
                    cfg.state_plan.recurrent,
                    new_recurrent_state[idx, t],
                )
                if carry_recurrent_scratch_ref is None:
                    vmem_ldst.store_state_region(*args)
                else:
                    pl.when(metadata_ref.p_id_is_last_tile[p_id, idx])(
                        functools.partial(vmem_ldst.store_state_region, *args)
                    )

    if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    raw_scratch_ref: dict[str, jax.Array] | None = None,
    zero_out_scratch_ref: tuple[jax.Array, jax.Array] | None = None,
    window_scratch_refs: tuple[jax.Array, ...] | None = None,
    *,
    cfg: config.GDNConfig,
):
    """Setup memory allocations and emit pipeline for running inner_kernel."""

    if cfg.state_plan is not None:
        # Both state regions stream from/to the single external source ref,
        # which rides the conv-state input/output slots.
        recurrent_state_ref = conv_state_ref
        recurrent_state_out_ref = conv_state_out_ref

    (
        qkv_alloc,
        b_alloc,
        a_alloc,
        conv_in_alloc,
        recurrent_in_alloc,
        out_alloc,
        conv_out_alloc,
        recurrent_out_alloc,
    ) = memory_ref.create_allocs(
        metadata_ref=metadata_ref,
        qkv_ref=qkv_ref,
        b_ref=b_ref,
        a_ref=a_ref,
        out_ref=out_ref,
        conv_state_ref=conv_state_ref,
        recurrent_state_ref=recurrent_state_ref,
        cfg=cfg,
    )

    num_tiles = metadata_ref.num_tiles[...]

    pipeline_func = pltpu.emit_pipeline(
        body=functools.partial(
            inner_kernel,
            cfg=cfg,
        ),
        grid=(num_tiles,),
        in_specs=(
            qkv_alloc.spec,
            b_alloc.spec,
            a_alloc.spec,
            conv_in_alloc.spec,
            recurrent_in_alloc.spec,
        ),
        out_specs=(
            out_alloc.spec,
            conv_out_alloc.spec,
            recurrent_out_alloc.spec,
        ),
    )

    @pl.with_scoped(
        allocations=(
            qkv_alloc,
            b_alloc,
            a_alloc,
            conv_in_alloc,
            recurrent_in_alloc,
            out_alloc,
            conv_out_alloc,
            recurrent_out_alloc,
        ),
    )
    def _run(allocations):
        pipeline_func(
            qkv_ref,
            b_ref,
            a_ref,
            conv_state_ref,
            recurrent_state_ref,
            out_ref,
            conv_state_out_ref,
            recurrent_state_out_ref,
            scratches=(
                metadata_ref,
                weights_ref,
                carry_conv_scratch_ref,
                carry_recurrent_scratch_ref,
                raw_scratch_ref,
            )
            + (() if window_scratch_refs is None else (window_scratch_refs,)),
            allocations=allocations,
        )

    if not cfg.kernel_zero_out:
        _run()
        return

    # Zero the output rows [first, batch_size) that no kernel of this call
    # writes (bucket padding, plus the PER_SEQ rows when PER_SEQ will not
    # run), in ZERO_OUT_ROWS-row DMAs overlapped with the pipeline. They are
    # disjoint from the rows the pipeline writes.
    zero_ref, zero_sem = cast(tuple[jax.Array, jax.Array], zero_out_scratch_ref)
    zero_ref[...] = jnp.zeros(zero_ref.shape, zero_ref.dtype)
    zero_rows = cfg.ZERO_OUT_ROWS
    first = metadata_ref.zero_out_query_start_loc[
        metadata_ref.zero_out_distribution[cfg.zero_out_from]
    ]
    num_copies = (cfg.batch_size - first + zero_rows - 1) // zero_rows

    def zero_copy(i):
        row = first + i * zero_rows
        size = jnp.minimum(zero_rows, cfg.batch_size - row)
        return pltpu.make_async_copy(
            zero_ref.at[pl.ds(0, size)], out_ref.at[pl.ds(row, size)], zero_sem
        )

    @pl.loop(0, num_copies)
    def _(i):
        zero_copy(i).start()

    _run()

    @pl.loop(0, num_copies)
    def _(i):
        zero_copy(i).wait()


# VLLM_TPU_GDN_V3_SINGLE_LAUNCH covers decode / verify programs only. In a
# prefill program (thousands of rows) the single-launch kernel next to a
# SparseCore-offloaded all-gather halts the step (c32, offload on: TC halt in
# the following adaptive_fused_moe, SC3 tile halt), while the two-call path
# runs clean. Decode steps never carry an all-gather above the 8 MiB offload
# threshold, and they are where the single launch saves its ~0.24 ms.
_SINGLE_LAUNCH_MAX_ROWS = 256


def _both_segments_kernel(
    metadata_refs: tuple[memory_ref.MetadataRef, memory_ref.MetadataRef],
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    source_ref: jax.Array,
    in_act_ref: jax.Array | None,
    weights_ref: memory_ref.WeightRefs,
    out_ref: jax.Array,
    source_out_ref: jax.Array,
    *,
    cfgs: tuple[config.GDNConfig, config.GDNConfig],
):
    """outer_kernel of the pooled BATCHED segment, then of the PER_SEQ one,
    on the same activations, output and state source
    (VLLM_TPU_GDN_V3_SINGLE_LAUNCH)."""
    for metadata_ref, cfg in zip(metadata_refs, cfgs, strict=True):

        def segment(metadata_ref=metadata_ref, cfg=cfg, **scratch_refs):
            outer_kernel(
                metadata_ref,
                qkv_ref,
                b_ref,
                a_ref,
                source_ref,
                None,
                in_act_ref,
                weights_ref,
                out_ref,
                source_out_ref,
                None,
                **scratch_refs,
                cfg=cfg,
            )

        pl.run_scoped(segment, **cfg.get_scratch_shape_dict())


def kernel_metadata(
    cfg: config.GDNConfig,
    *,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    distribution: jax.Array,
    ckpt_indices: jax.Array | None,
    per_seq_inputs: tuple[jax.Array, ...] | None = None,
    prefill_ckpt: tuple[jax.Array, jax.Array] | None = None,
    tile_sized: bool = False,
) -> memory_ref.MetadataRef:
    """The scalar-prefetch metadata of one `fused_conv1d_gdn` kernel call.

    `tile_sized`: PER_SEQ per-tile arrays sized by tiles, not tokens
    (`metadata.per_seq_tile_slots`)."""
    if cfg.mode != config.GDNMode.PER_SEQ:
        return metadata.compute_batched_seq_metadata(
            cfg=cfg,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            end_seq=distribution[0],
            ckpt_indices=ckpt_indices,
        )
    if per_seq_inputs is not None:
        ps_qsl, ps_seq_lens, ps_state, ps_read, ps_ckpt = per_seq_inputs
        return metadata.compute_per_seq_metadata(
            cfg=cfg,
            seq_lens=ps_seq_lens,
            query_start_loc=ps_qsl,
            state_indices=ps_state,
            read_offsets=ps_read,
            start_seq=distribution[0],
            end_seq=distribution[-1],
            ckpt_indices=ps_ckpt,
            prerolled=True,
            prefill_ckpt=prefill_ckpt,
            tile_sized=tile_sized,
        )
    return metadata.compute_per_seq_metadata(
        cfg=cfg,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        read_offsets=read_offsets,
        start_seq=distribution[0],
        end_seq=distribution[-1],
        ckpt_indices=ckpt_indices,
        prefill_ckpt=prefill_ckpt,
        tile_sized=tile_sized,
    )


def pack_kernel_metadata(metas: tuple[memory_ref.MetadataRef, ...]) -> jax.Array:
    """Every leaf of `metas`, flattened in order, as one int32 vector.

    All MetadataRef leaves are int32 or bool scalars / arrays, so the round
    trip through `unpack_kernel_metadata` is exact.
    """
    leaves = jax.tree.leaves(metas)
    for leaf in leaves:
        dtype = jnp.result_type(leaf)
        if not (jnp.issubdtype(dtype, jnp.integer) or dtype == jnp.bool_):
            raise TypeError(f"GDN kernel metadata leaf of dtype {dtype}")
    return jnp.concatenate([jnp.ravel(x).astype(jnp.int32) for x in leaves])


def packed_kernel_metadata_size(
    max_tokens: int, max_seqs: int, num_spec_tokens: int
) -> int:
    """Static upper bound on `pack_kernel_metadata`'s length for calls of at
    most `max_tokens` rows and `max_seqs` sequence slots.

    Depends on no traced shape, so the op returning it keeps a static output
    shape under a dynamic token dim. Per segment: `num_tiles`, up to six
    per-tile arrays (at most one entry per token, or per slot), the
    per-sequence arrays incl. the checkpoint ids, and the zero-fill
    `query_start_loc` / `distribution`.
    """
    per_tile = max(max_tokens, max_seqs) + 2
    per_seq = max_seqs * (num_spec_tokens + 4) + (max_seqs + 1) + 3
    return 2 * (1 + 6 * per_tile + per_seq)


def unpack_kernel_metadata(
    packed: jax.Array, abstract: tuple[memory_ref.MetadataRef, ...]
) -> tuple[memory_ref.MetadataRef, ...]:
    """Inverse of `pack_kernel_metadata` for metadata shaped like `abstract`
    (`jax.eval_shape` of the same builders)."""
    leaves, treedef = jax.tree.flatten(abstract)
    sizes = [math.prod(leaf.shape) for leaf in leaves]
    # A static-size vector (packed_size) carries zero padding past the end.
    if packed.ndim != 1 or packed.shape[0] < sum(sizes):
        raise ValueError(
            f"packed GDN kernel metadata {packed.shape} < ({sum(sizes)},)"
        )
    out, offset = [], 0
    for leaf, size in zip(leaves, sizes, strict=True):
        part = packed[offset : offset + size].reshape(leaf.shape)
        out.append(part.astype(leaf.dtype))
        offset += size
    return jax.tree.unflatten(treedef, out)


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state", "state_source"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "num_spec_tokens",
        "conv_weight_kmajor",
        "decode_tile_size",
        "mixed_tile_size",
        "batched_only",
        "prefill_only",
        "zero_initialize_out",
        "compute_precision",
        "state_plan",
        "attention_mode",
        "gate_lower_bound",
        "prefill_ckpt_alignment",
        "metadata_only",
        "packed_size",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array | Conv1DInput,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
    b: jax.Array,  # [batch_size, n_v]
    a: jax.Array,  # [batch_size, n_v]; [batch_size, n_v * d_k] for KDA
    conv_state: jax.Array | None,  # [num_seqs + 1, kernel_size - 1, dim_size]
    recurrent_state: jax.Array | None,  # [num_seqs + 1, nv, dk, dv]
    conv_weight: jax.Array,  # [kernel_size - 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v] for GDN, [n_v, d_k] for KDA
    query_start_loc: jax.Array,  # [num_seqs + 1]
    state_indices: jax.Array,  # [num_seqs]
    distribution: jax.Array,  # [3]
    seq_lens: jax.Array,  # [num_seqs]
    read_offsets: jax.Array | None = None,  # [num_seqs]
    ckpt_indices: jax.Array | None = None,  # [num_seqs, num_spec_tokens + 1]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    gate_lower_bound: float | None = None,
    state_source: jax.Array | None = None,
    state_plan: config.StateSourcePlan | None = None,
    num_spec_tokens: int = 0,
    conv_weight_kmajor: bool = False,
    batched_only: bool = False,
    prefill_only: bool = False,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    # TODO(kyuyeunk): Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
    per_seq_inputs: tuple[jax.Array, ...] | None = None,
    prefill_ckpt: tuple[jax.Array, jax.Array] | None = None,
    prefill_ckpt_alignment: int = 0,
    packed_metadata: jax.Array | None = None,
    metadata_only: bool = False,
    packed_size: int = 0,
) -> (
    tuple[tuple[jax.Array, jax.Array], jax.Array]
    | tuple[jax.Array, jax.Array]
    | jax.Array
):
    """Perform conv1d and gdn in a single fused kernel.

    Args:
        qkv: Mixed query, key, value input tensor of shape [batch_size, dim_size],
            where `dim_size = n_kq * d_k * 2 + n_v * d_v`, or a prepared
            Conv1DInput with FP32 values and the original output dtype.
        b: b tensor (for beta) of shape [batch_size, n_v].
        a: a tensor (for g) of shape [batch_size, n_v].
        conv_state: Convolution state cache tensor of shape [num_seqs + 1,
            kernel_size - 1, dim_size] containing the last (kernel_size - 1) tokens
            from the last sequence invocation. The first slot is a null block used for
            padded or invalid tokens. It may contain garbage data if it is a first
            invocation of a sequence. None when `state_source` carries the states.
        recurrent_state: Recurrent state cache tensor of shape [num_seqs + 1, n_v,
            d_k, d_v]. The first slot is a null block used for padded or invalid
            tokens. It may contain garbage data if it is a first invocation of a
            sequence. None when `state_source` carries the states.
        conv_weight: Convolution weight tensor of shape [kernel_size - 1, dim_size].
        conv_bias: Optional convolution bias tensor of shape [dim_size].
        a_log: a_log tensor of shape [n_v].
        dt_bias: dt_bias tensor of shape [n_v]; [n_v, d_k] under KDA, whose
            gate is per-channel. Kept 2D rather than flattened because the
            KDA path reshapes it to broadcast over (head, channel), and
            Mosaic rejects that shape cast from a flat vector
            ("infer-vector-layout: unsupported shape cast").
        query_start_loc: Start locations of sequences of shape [num_seqs + 1].
        state_indices: Indices mapping sequences to state cache slots of shape
            [num_seqs]. With speculative decoding these are the *base* slots
            of per-request groups of `num_spec_tokens + 1` consecutive slots.
        distribution: Tensor of shape [3] int32 — [decode_end, prefill_end,
            mixed_end]. With `num_spec_tokens > 0`, the first segment holds
            speculative verify windows of up to `num_spec_tokens + 1` tokens
            rather than 1-token decodes.
        seq_lens: Sequence lengths for each sequence of shape [num_seqs].
        read_offsets: Optional [num_seqs] int32 — per-sequence state read
            offset (num_accepted - 1 from the last verify step). Windowed
            sequences read their initial state from
            `state_indices[s] + read_offsets[s]` and write one checkpoint
            per window position to `state_indices[s] + t`. Required when
            `num_spec_tokens > 0`.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key/query dimension.
        d_v: Value dimension.
        kernel_size: Convolution kernel size.
        attention_mode: `GDN` for the per-head scalar gate, `KDA` for the
            per-channel one. Selects which compute path runs.
        gate_lower_bound: KDA only, and must be negative. `None` gives the
            unbounded gate `-exp(A_log) * softplus(g + dt_bias)`
            (Kimi-Linear-48B); a float gives the bounded
            `gate_lower_bound * sigmoid(exp(A_log) * (g + dt_bias))` (every K3
            variant, at -5.0). Different functions of the same weights, so a
            mismatch with the model's KDA config silently changes the decay.
        state_source: Optional indexed external state source (e.g. the unified
            KV pool) replacing the dense state tensors: both state regions are
            streamed between it and the kernel pipeline per `state_plan`, with
            `state_indices` addressing source block windows. Donated and
            returned in place of the dense states.
        state_plan: Static copy-plan describing the source's state regions.
            Must be set iff `state_source` is set.
        num_spec_tokens: Number of speculative draft tokens (0 gives the
            plain 1-token-per-sequence decode path).
        batched_only: Run only the leading decode/verify segment.
        prefill_only: Run only PER_SEQ, leaving decode/verify checkpoints
            untouched. Mutually exclusive with batched_only.
        zero_initialize_out: Whether to zero-initialize the output buffer before
            executing non-batched sequences.
        compute_precision: Computation precision dtype.
        decode_tile_size: Tile size along sequence dimension for decode sequences.
        per_seq_inputs: Optional `metadata.roll_per_seq_inputs(query_start_loc,
            seq_lens, state_indices, read_offsets, distribution[0],
            ckpt_indices)` (read offsets and checkpoint ids as int32), rolled
            once per step by the caller instead of once per layer here.
            Ignored when TPU_ENABLE_GDN_DYNAMIC_TILING is set.
        mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
            sequences. Ignored when TPU_ENABLE_GDN_DYNAMIC_TILING is set.
        prefill_ckpt: Optional `(offsets, blocks)`, int32 [num_seqs] each:
            internal prefill checkpoints (VLLM_TPU_MAMBA_TAIL_FOLD). A PER_SEQ
            sequence also writes its state after its first `offsets[s]` query
            tokens to checkpoint block `blocks[s]`; 0 means none. See
            `metadata.compute_per_seq_metadata`.
        prefill_ckpt_alignment: Every nonzero checkpoint offset is a multiple
            of this; the PER_SEQ tile must divide it.
        packed_metadata: Optional int32 vector from a `metadata_only` call
            with the same shapes and step inputs (VLLM_TPU_GDN_STEP_KERNEL_METADATA):
            both segments' kernel metadata, built once per step and shared by
            every layer, replacing the per-call metadata computation. Pooled
            calls running both segments only.
        metadata_only: Return the call's packed kernel metadata (see
            `pack_kernel_metadata`) instead of running the kernel.
        packed_size: With `metadata_only`, zero-pad the packed vector to this
            static length (`packed_kernel_metadata_size`), so its shape does
            not depend on the token count of the call; 0 keeps it exact.

    Returns:
        (new_conv_state, new_recurrent_state): Updated convolution state cache and
            recurrent state cache tensors. With a state source: the updated
            source (aliased in place) instead of the state pair.
        out: Fused output tensor.
    """
    step_metadata = packed_metadata is not None or metadata_only
    if step_metadata and (prefill_ckpt is not None or per_seq_inputs is not None):
        raise ValueError(
            "packed GDN kernel metadata excludes prefill_ckpt and per_seq_inputs"
        )
    # Conv1D consumes FP32 while outputs retain the projection activation dtype.
    prepared_qkv = isinstance(qkv, Conv1DInput)
    if isinstance(qkv, Conv1DInput):
        act_out_dtype = qkv.output_dtype
        qkv = qkv.values
        if qkv.ndim != 3 or qkv.shape[1] != 1 or qkv.dtype != jnp.float32:
            raise ValueError("Conv1DInput requires FP32 [tokens, 1, features]")
    else:
        act_out_dtype = qkv.dtype
    pooled = state_plan is not None
    if pooled:
        assert conv_state is None and recurrent_state is None
        assert state_source is not None
        # Checkpoints are individually addressable source blocks; without
        # the index array they would all alias onto the slot's own block.
        assert num_spec_tokens == 0 or ckpt_indices is not None, (
            "num_spec_tokens > 0 requires ckpt_indices"
        )
    else:
        assert state_source is None
        assert conv_state is not None and recurrent_state is not None
        conv_out_dtype = conv_state.dtype
        recurrent_out_dtype = recurrent_state.dtype
        conv_state = conv_state.astype(jnp.float32)

    # Step 1: Validate inputs.
    is_kda = attention_mode == config.AttentionMode.KDA

    # Raw inputs: the kernel DMAs tile-aligned row blocks of the unpadded 2D
    # projection outputs and casts in VMEM (see RowBlockInBufferedRef), so
    # the token count must be a whole number of HBM row tiles.
    def raw_row_align(x: jax.Array) -> int:
        itemsize = jnp.dtype(x.dtype).itemsize
        return 8 * (4 // itemsize) if itemsize in (2, 4) else 0

    def raw_eligible(x: jax.Array) -> bool:
        align = raw_row_align(x)
        return x.ndim == 2 and align > 0 and x.shape[0] % align == 0

    use_raw_qkv = (
        envs.TPU_GDN_RAW_QKV and not prepared_qkv and raw_eligible(qkv)
    )
    use_raw_ba = (
        envs.TPU_GDN_RAW_BA
        and not is_kda
        and raw_eligible(b)
        and raw_eligible(a)
        and b.shape[0] == a.shape[0] == qkv.shape[0]
        and raw_row_align(b) == raw_row_align(a)
    )
    use_kernel_zero_out = (
        envs.TPU_GDN_KERNEL_ZERO_OUT and zero_initialize_out and not prefill_only
    )
    if use_raw_qkv or use_raw_ba or use_kernel_zero_out:
        _key = (use_raw_qkv, use_raw_ba, use_kernel_zero_out, qkv.shape[0])
        if _key not in _RAW_PATH_LOGGED:
            _RAW_PATH_LOGGED.add(_key)
            logger.info(
                "GDN_RAW_PATH raw_qkv=%s raw_ba=%s kernel_zero_out=%s rows=%d",
                *_key,
            )

    if not prepared_qkv and not use_raw_qkv:
        qkv = qkv.astype(jnp.float32)
    if not use_raw_ba:
        b = b.astype(jnp.float32)
        a = a.astype(jnp.float32)
    if gate_lower_bound is not None:
        if not is_kda:
            raise ValueError(
                "gate_lower_bound only applies to KDA; got "
                f"attention_mode={attention_mode}."
            )
        # The chunked solve needs every causal exponent g_r - g_t to stay
        # non-positive, which `gate_lower_bound * sigmoid(.)` only satisfies
        # for a negative bound.
        if gate_lower_bound >= 0:
            raise ValueError(
                "gate_lower_bound must be negative (it bounds a log-decay); "
                f"got {gate_lower_bound}. A non-negative bound makes the gate "
                "increase along the chunk, which this kernel does not "
                "support."
            )

    num_seqs = state_indices.size
    batch_size, dim = qkv.shape[0], qkv.shape[-1]
    if conv_weight_kmajor:
        # Already float32 in the kernel's [kernel_size, 1, dim] order.
        assert conv_weight.shape == (kernel_size, 1, dim)
        assert conv_weight.dtype == jnp.float32
    else:
        assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim,)
    assert query_start_loc.shape == (num_seqs + 1,)
    assert state_indices.shape == (num_seqs,)
    assert distribution.shape == (3,)
    if num_spec_tokens > 0:
        assert read_offsets is not None, (
            "read_offsets is required when num_spec_tokens > 0"
        )
    if read_offsets is None:
        read_offsets = jnp.zeros((num_seqs,), dtype=jnp.int32)
    assert read_offsets.shape == (num_seqs,)
    read_offsets = read_offsets.astype(jnp.int32)
    if ckpt_indices is not None:
        # One source block per checkpoint (vLLM's ssm_state_indices scheme):
        # checkpoint t of sequence s lives at block ckpt_indices[s, t], so
        # the pool block only ever has to hold a single state.
        assert ckpt_indices.shape == (num_seqs, num_spec_tokens + 1), (
            ckpt_indices.shape,
            num_seqs,
            num_spec_tokens,
        )
        ckpt_indices = ckpt_indices.astype(jnp.int32)
    act_in_dtype = jnp.dtype(jnp.float32)
    assert use_raw_ba or a.dtype == b.dtype == act_in_dtype

    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    decode_tile_size = min(decode_tile_size, batch_size)
    mixed_tile_size = min(mixed_tile_size, batch_size)
    if envs.VLLM_TPU_GDN_MIXED_TILE_POW2 and mixed_tile_size > 1:
        # Round the static clip down to a power of two so a non-power-of-two
        # token bucket (24, 40 rows) stays divisible by the triangular
        # solver's block size (port of PR #2821 ece3d83b5 / PR #2438).
        # Dynamic tiling already picks powers of two.
        mixed_tile_size = 1 << (mixed_tile_size.bit_length() - 1)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes
    window_size = num_spec_tokens + 1
    decode_vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(window_size)

    # KDA stays on static tiles until the VMEM estimate models its gate.
    if envs.TPU_ENABLE_GDN_DYNAMIC_TILING and not is_kda:
        decode_tile_size, mixed_tile_size = tiling.get_tile_sizes(
            batch_size=batch_size,
            num_seqs=num_seqs,
            padded_batch_size=padded_batch_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim if conv_state is None else conv_state.shape[-1],
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=(
                jnp.float32 if recurrent_state is None else recurrent_state.dtype
            ),
            num_lanes=num_lanes,
            decode_vmem_limit_bytes=decode_vmem_limit_bytes,
            mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
            window_size=window_size,
        )
        logger.info_once(
            "GDN dynamic tiling: num_seqs=%d batch=%d window=%d -> T=%d C=%d",
            num_seqs,
            batch_size,
            window_size,
            decode_tile_size,
            mixed_tile_size,
        )
    else:
        # Each sequence holds 1 input state slot plus `window` output
        # checkpoint slots per buffer in VMEM. Shrink the tile so the
        # double-buffered windows fit in roughly half the scoped-VMEM budget.
        window = num_spec_tokens + 1
        num_buffers = config.GDNConfig.__dataclass_fields__["num_buffers"].default
        state_positions = 1 + window
        recurrent_bytes_per_seq = state_positions * n_v * d_k * d_v * 4
        bytes_per_seq = (
            # Recurrent checkpoints (fp32) — the dominant term.
            recurrent_bytes_per_seq
            # Conv checkpoints (fp32).
            + state_positions * (kernel_size - 1) * dim * 4
            # qkv (fp32), b/a (fp32), out (act_out).
            + window * (dim * 4 + 2 * aligned_num_v_heads * 4 + n_v * d_v * 2)
        )
        spec_tile_budget = (decode_vmem_limit_bytes // 2) // num_buffers
        decode_tile_size = max(
            1, min(decode_tile_size, spec_tile_budget // bytes_per_seq)
        )

    if num_spec_tokens > 0 and is_kda:
        # KDA verify keeps one sequence per tile until multi-sequence KDA
        # verify tiles are sized by a KDA-aware VMEM estimate.
        decode_tile_size = 1

    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    # KDA's gate carries d_k channels per value head, so it pads out to a
    # lane multiple of n_v * d_k rather than of n_v.
    gate_dim = n_v * d_k if is_kda else n_v
    gate_padding_size = (pl.cdiv(gate_dim, num_lanes) * num_lanes) - gate_dim
    # FP32 activations need no batch padding (packing is 1); the raw path
    # relies on that to read the unpadded inputs with batch_size rows.
    assert not (use_raw_qkv or use_raw_ba) or padded_batch_size == batch_size
    if not use_raw_qkv:
        qkv_padding = ((0, batch_padding_size),) + ((0, 0),) * (qkv.ndim - 1)
        qkv = jnp.pad(qkv, qkv_padding)
        qkv = qkv.reshape(padded_batch_size, 1, -1)
    if not use_raw_ba:
        b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
        a = jnp.pad(a, ((0, batch_padding_size), (0, gate_padding_size)))
        b = b.reshape(padded_batch_size, 1, -1)
        a = a.reshape(padded_batch_size, 1, -1)

    # Step 3: States and weights pre-processing.
    # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
    # loading stage.
    if not pooled:
        conv_state_shape = cast(jax.Array, conv_state).shape
        conv_state = cast(jax.Array, conv_state).reshape(-1, kernel_size - 1, 1, dim)
    if not conv_weight_kmajor:
        conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = conv_bias.astype(jnp.float32) if conv_bias is not None else None

    # Step 4: Wrap inputs for the kernel.
    conv_weights = memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias)
    gdn_weights = memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
    weights = memory_ref.WeightRefs(conv=conv_weights, gdn=gdn_weights)

    # Step 5: Create specs.
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    def make_cfg_and_metadata(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array | None,
        in_act: jax.Array | None,
        mode: config.GDNMode,
    ) -> tuple[config.GDNConfig, memory_ref.MetadataRef]:
        # With a state source, it rides the conv-state slot and the dense
        # recurrent-state input is absent.
        chunk_pair = False
        if mode == config.GDNMode.PER_SEQ:
            tile_size = mixed_tile_size
            # Prefill / mixed sequences keep a single state checkpoint.
            window_size = 1
            # Two chunks per grid step (chunked GDN only; chunk 1 is the
            # token-recurrent scan).
            if (
                envs.TPU_GDN_CHUNK_PAIR
                and not is_kda
                and tile_size > 1
                and 2 * tile_size <= padded_batch_size
            ):
                tile_size, chunk_pair = 2 * tile_size, True
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1

        # Rows a tile's tokens span (seq_tile_size * chunk_size, both modes).
        tile_rows = tile_size * (1 if mode == config.GDNMode.PER_SEQ else window_size)

        def raw_rows(x: jax.Array) -> int:
            # A block starting up to align - 1 rows before the tile's first
            # token still covers all of its tokens.
            align = raw_row_align(x)
            return min(pl.cdiv(tile_rows, align) * align + align, batch_size)

        # The first kernel of the call zero-fills the rows no kernel writes:
        # those past the PER_SEQ segment, or past the BATCHED one when
        # PER_SEQ does not run.
        zero_out_from = -1
        if use_kernel_zero_out and in_act is None:
            zero_out_from = 0 if batched_only else 2

        cfg = config.GDNConfig(
            mode=mode,
            batch_size=padded_batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            window_size=window_size,
            dim_size=dim,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d_k,
            v_head_dim=d_v,
            attention_mode=attention_mode,
            gate_lower_bound=gate_lower_bound,
            chunk_pair=chunk_pair,
            dtypes=config.Dtypes(
                act_in=act_in_dtype,
                act_out=act_out_dtype,
                compute=compute_precision,
                recurrent_state=(
                    jnp.float32.dtype
                    if pooled
                    else cast(jax.Array, in_recurrent_state).dtype
                ),
                conv_state=(jnp.float32.dtype if pooled else in_conv_state.dtype),
            ),
            state_plan=state_plan,
            raw_qkv_rows=raw_rows(qkv) if use_raw_qkv else 0,
            raw_qkv_align=raw_row_align(qkv) if use_raw_qkv else 0,
            raw_ba_rows=raw_rows(b) if use_raw_ba else 0,
            raw_ba_align=raw_row_align(b) if use_raw_ba else 0,
            zero_out_from=zero_out_from,
        )
        if (
            envs.VLLM_TPU_GDN_V3_WINDOW_VERIFY
            and pooled
            and mode == config.GDNMode.BATCHED
            and window_size > 1
            and not is_kda
        ):
            window_cfg = dataclasses.replace(cfg, window_verify=True)
            if window_tile.supported(
                window_cfg, in_conv_state.shape, in_conv_state.dtype
            ):
                cfg = window_cfg
            else:
                logger.info_once(
                    "GDN window verify: pool layout not supported by "
                    "window_tile; verify keeps inner_kernel"
                )

        # Step 6: Metadata preprocessing. Runs once per layer: each layer's
        # op is its own program, so XLA does not share it across layers
        # (pass `per_seq_inputs` to hoist the PER_SEQ roll).
        ckpt = prefill_ckpt if mode == config.GDNMode.PER_SEQ else None
        if ckpt is not None:
            if prefill_ckpt_alignment <= 0:
                raise ValueError("prefill_ckpt needs a prefill_ckpt_alignment")
            # A checkpoint needs a query longer than one alignment unit, so
            # smaller buckets (and their odd tile sizes) never carry one.
            if padded_batch_size <= prefill_ckpt_alignment:
                ckpt = None
            elif prefill_ckpt_alignment % cfg.chunk_size:
                raise ValueError(
                    f"GDN prefill tile {cfg.chunk_size} does not divide the "
                    f"checkpoint alignment {prefill_ckpt_alignment}"
                )
        metadata_obj = kernel_metadata(
            cfg,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            distribution=distribution,
            ckpt_indices=ckpt_indices,
            per_seq_inputs=per_seq_inputs,
            prefill_ckpt=ckpt,
            tile_sized=step_metadata,
        )

        if cfg.kernel_zero_out:
            metadata_obj = dataclasses.replace(
                metadata_obj,
                zero_out_query_start_loc=query_start_loc,
                zero_out_distribution=distribution,
            )
        return cfg, metadata_obj

    def call_kernel(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array | None,
        in_act: jax.Array | None,
        mode: config.GDNMode,
        metadata_override: memory_ref.MetadataRef | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        cfg, metadata_obj = make_cfg_and_metadata(
            in_conv_state, in_recurrent_state, in_act, mode
        )
        if metadata_override is not None:
            metadata_obj = metadata_override
        metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)

        # Step 7: Handle case where write needs to be done in existing out.
        # Flat aliasing indices account for the absent recurrent-state
        # input in source mode (None contributes no leaves).
        in_out_spec = None
        num_state_inputs = 1 if pooled else 2
        input_output_aliases = {len(metadata_obj) + 3: 1}
        if not pooled:
            input_output_aliases[len(metadata_obj) + 4] = 2
        out_shape = cfg.get_out_shape()

        if in_act is None and zero_initialize_out and not cfg.kernel_zero_out:
            in_act = jnp.zeros_like(out_shape)
        if in_act is not None:
            out_shape = in_act
            in_out_spec = hbm_spec
            input_output_aliases[len(metadata_obj) + 3 + num_state_inputs] = 0

        vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(cfg.window_size)

        return pl.pallas_call(
            functools.partial(outer_kernel, cfg=cfg),
            out_shape=(out_shape, in_conv_state, in_recurrent_state),
            in_specs=(
                metadata_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                None if pooled else hbm_spec,
                in_out_spec,
                weights_spec,
            ),
            out_specs=(hbm_spec, hbm_spec, None if pooled else hbm_spec),
            scratch_shapes=cfg.get_scratch_shape_dict(),
            input_output_aliases=input_output_aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            name=cfg.get_kernel_name(),
            metadata=cfg.get_metadata(),
        )(
            metadata_obj,
            qkv,
            b,
            a,
            in_conv_state,
            in_recurrent_state,
            in_act,
            weights,
        )

    if batched_only and prefill_only:
        raise ValueError("batched_only and prefill_only are mutually exclusive")

    # VLLM_TPU_GDN_STEP_KERNEL_METADATA: both segments' metadata as one packed
    # int32 vector, built by a `metadata_only` call once per step.
    segment_metas = None
    if step_metadata:
        if not pooled or prefill_only or batched_only:
            raise ValueError(
                "packed GDN kernel metadata covers pooled calls running both "
                "segments only"
            )
        source = cast(jax.Array, state_source)

        def build(mode: config.GDNMode, in_act):
            return make_cfg_and_metadata(source, None, in_act, mode)[1]

        batched_build = functools.partial(build, config.GDNMode.BATCHED, None)
        batched_out = make_cfg_and_metadata(
            source, None, None, config.GDNMode.BATCHED
        )[0].get_out_shape()
        per_seq_build = functools.partial(build, config.GDNMode.PER_SEQ, batched_out)
        if metadata_only:
            packed = pack_kernel_metadata((batched_build(), per_seq_build()))
            if packed_size:
                if packed.size > packed_size:
                    raise ValueError(
                        f"GDN kernel metadata needs {packed.size} entries, "
                        f"more than the static bound {packed_size}"
                    )
                packed = jnp.pad(packed, (0, packed_size - packed.size))
            return packed
        segment_metas = unpack_kernel_metadata(
            cast(jax.Array, packed_metadata),
            (jax.eval_shape(batched_build), jax.eval_shape(per_seq_build)),
        )

    if (
        pooled
        and envs.VLLM_TPU_GDN_V3_SINGLE_LAUNCH
        and not (prefill_only or batched_only)
        and batch_size <= _SINGLE_LAUNCH_MAX_ROWS
    ):
        # Both segments in one kernel (VLLM_TPU_GDN_V3_SINGLE_LAUNCH), each
        # with the configuration, metadata and scratch it has as its own
        # call; the BATCHED segment owns the in-kernel zero-fill.
        source = cast(jax.Array, state_source)
        batched_cfg, batched_meta = make_cfg_and_metadata(
            source, None, None, config.GDNMode.BATCHED
        )
        if segment_metas is not None:
            batched_meta = segment_metas[0]
        in_act = None
        if zero_initialize_out and not batched_cfg.kernel_zero_out:
            in_act = jnp.zeros_like(batched_cfg.get_out_shape())
        # The PER_SEQ segment writes into the BATCHED segment's output, as
        # its own call does through the aliased `in_act`.
        per_seq_cfg, per_seq_meta = make_cfg_and_metadata(
            source,
            None,
            batched_cfg.get_out_shape(),
            config.GDNMode.PER_SEQ,
        )
        if segment_metas is not None:
            per_seq_meta = segment_metas[1]
        cfgs = (batched_cfg, per_seq_cfg)
        metas = (batched_meta, per_seq_meta)
        num_meta = len(batched_meta) + len(per_seq_meta)
        aliases = {num_meta + 3: 1}
        if in_act is not None:
            aliases[num_meta + 4] = 0
        out_act, out_source = pl.pallas_call(
            functools.partial(_both_segments_kernel, cfgs=cfgs),
            out_shape=(batched_cfg.get_out_shape(), source),
            in_specs=(
                jax.tree.map(lambda _: smem_spec, metas),
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                None if in_act is None else hbm_spec,
                weights_spec,
            ),
            out_specs=(hbm_spec, hbm_spec),
            input_output_aliases=aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=max(
                    config.GDNConfig.get_vmem_limit_bytes(c.window_size) for c in cfgs
                ),
            ),
            name=f"{batched_cfg.get_kernel_name()}_and_per_seq",
            metadata=batched_cfg.get_metadata(),
        )(metas, qkv, b, a, source, in_act, weights)
        out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
        return out_source, out_act

    if pooled:
        out_act, out_source = None, state_source
        if not prefill_only:
            out_act, out_source, _ = call_kernel(
                state_source,
                None,
                None,
                config.GDNMode.BATCHED,
                None if segment_metas is None else segment_metas[0],
            )
        if not batched_only:
            out_act, out_source, _ = call_kernel(
                out_source,
                None,
                out_act,
                config.GDNMode.PER_SEQ,
                None if segment_metas is None else segment_metas[1],
            )
        out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
        return out_source, out_act

    # The first segment holds verify windows of up to `num_spec_tokens + 1`
    # tokens, or plain 1-token decodes without speculative decoding.
    out_act, out_conv_state, out_recurrent_state = None, conv_state, recurrent_state
    if not prefill_only:
        out_act, out_conv_state, out_recurrent_state = call_kernel(
            conv_state, recurrent_state, None, config.GDNMode.BATCHED
        )
    if not batched_only:
        out_act, out_conv_state, out_recurrent_state = call_kernel(
            out_conv_state, out_recurrent_state, out_act, config.GDNMode.PER_SEQ
        )

    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    out_conv_state = out_conv_state.astype(conv_out_dtype)
    out_conv_state = out_conv_state.reshape(conv_state_shape)
    out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

    return (out_conv_state, out_recurrent_state), out_act
