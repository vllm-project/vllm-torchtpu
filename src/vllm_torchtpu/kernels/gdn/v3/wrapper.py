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
)
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput, QKVColumns
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _unpack_qkv_2d(
    qkv_slot_ref: jax.Ref,
    act_scratch_ref: jax.Ref,
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    cfg: config.GDNConfig,
) -> jax.Ref:
    """Casts a tile's 2D bf16 qkv rows to f32 compact rows in VMEM.

    `memory_ref.qkv_2d_rows` DMAs whole 16-row tiles, so the tile's first
    row sits `r_base % 16` rows into the window. Each of the 16 possible
    offsets gets a static branch, shifting the rows out of their aligned
    load into `act_scratch_ref` ([seq, aligned_chunk, 1, dim] f32). Rows past
    the tile's real size carry neighbouring tokens or stale VMEM, like the
    f32 window's do; the kernel masks them by `real_sizes`.

    Returns:
        The f32 [seq, chunk, 1, dim] ref the rest of the kernel reads.
    """
    align = config.QKV_2D_ROW_ALIGN
    aligned_chunk = cfg.qkv_2d_aligned_chunk
    for idx in range(cfg.seq_tile_size):
        r_prefix = jax.lax.rem(metadata_ref.p_id_to_r_base[p_id, idx], align)
        for prefix in range(align):

            @pl.when(r_prefix == prefix)
            def _unpack(idx=idx, prefix=prefix):
                load_rows = pl.cdiv(prefix + aligned_chunk, align) * align
                rows = qkv_slot_ref[idx, :load_rows, :]
                act_scratch_ref[idx, :aligned_chunk, 0, :] = rows[
                    prefix : prefix + aligned_chunk, :
                ].astype(jnp.float32)

    if aligned_chunk == cfg.chunk_size:
        return act_scratch_ref
    return act_scratch_ref.at[:, : cfg.chunk_size]


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size]; 2D bf16: [seq, rows, dim]
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
    act_scratch_ref: jax.Ref | None = None,
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
        act_scratch_ref: With `cfg.qkv_2d`, the f32 VMEM scratch the 2D bf16
            qkv rows are cast into; it then stands in for `qkv_slot_ref`.
        cfg: GDN configuration object.
        p_id: Optional explicit tile index. Communication-fused outer kernels
            pass their global compute-tile index because their surrounding
            pipeline owns the Pallas program id.
    """

    if p_id is None:
        p_id = pl.program_id(0)

    if cfg.dense_conv:
        _dense_inner_kernel(
            qkv_slot_ref=qkv_slot_ref,
            b_slot_ref=b_slot_ref,
            a_slot_ref=a_slot_ref,
            conv_state_in_slot_ref=conv_state_in_slot_ref,
            recurrent_in_slot_ref=recurrent_in_slot_ref,
            out_slot_ref=out_slot_ref,
            conv_state_out_slot_ref=conv_state_out_slot_ref,
            recurrent_out_slot_ref=recurrent_out_slot_ref,
            metadata_ref=metadata_ref,
            weights_ref=weights_ref,
            carry_conv_scratch_ref=carry_conv_scratch_ref,
            carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
            dense_ref=act_scratch_ref,
            cfg=cfg,
            p_id=p_id,
        )
        return

    qkv_work_ref = qkv_slot_ref
    if cfg.qkv_2d:
        # Unpack before the states are selected, so the bf16 load and f32
        # cast temporaries are dead before the selected states become live
        # (4 MiB of VMEM peak for the PER_SEQ b1 c64 kernel at dim 12288 on
        # TPU7x when ordered after, measured on the Qwen3.5 branch).
        assert cfg.mode == config.GDNMode.PER_SEQ and act_scratch_ref is not None
        qkv_work_ref = _unpack_qkv_2d(
            qkv_slot_ref, act_scratch_ref, metadata_ref, p_id, cfg
        )

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
    qkv_in_compact = qkv_work_ref[...].astype(jnp.float32)
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
                qkv_vmem_ref=qkv_work_ref,
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
                qkv_vmem_ref=qkv_work_ref,
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


def _dense_inner_kernel(
    *,
    qkv_slot_ref: jax.Ref,  # [1, aligned_chunk + 16, dim] bf16
    b_slot_ref: jax.Ref,
    a_slot_ref: jax.Ref,
    conv_state_in_slot_ref: jax.Ref,
    recurrent_in_slot_ref: jax.Ref,
    out_slot_ref: jax.Ref,
    conv_state_out_slot_ref: jax.Ref,
    recurrent_out_slot_ref: jax.Ref,
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Ref,
    carry_recurrent_scratch_ref: jax.Ref,
    dense_ref: jax.Ref,  # [8 + aligned_chunk, dim] f32
    cfg: config.GDNConfig,
    p_id: jax.Array,
) -> None:
    """PER_SEQ chunked tile over 2D bf16 rows on token-dense rows
    (VLLM_TPU_GDN_PREFILL_DENSE_CONV; Qwen3.5's TPU_GDN_PREFILL_DENSE_CONV).

    The compact path keeps one token row per vreg (one of eight sublanes in
    use), so each conv tap is a whole-vreg op per 128 lanes per token, and it
    stages the activations through VMEM for strided head loads. Here the
    tile's rows are unpacked dense, a tap is a sublane shift of one load, and
    each head is a lane range of the activated rows. The arithmetic is the
    compact path's, op for op: the conv accumulates from zero in tap order,
    then the bias, then SiLU; the conv state is the same masked row pick.
    """
    chunk, kprev, dim = cfg.chunk_size, cfg.prev_kernel_size, cfg.dim_size
    align = config.QKV_2D_ROW_ALIGN
    aligned_chunk = cfg.qkv_2d_aligned_chunk

    # Unpack the tile's rows out of their 16-row aligned DMA window (before the
    # states are selected, as the compact 2D path does, to keep the VMEM peak).
    r_prefix = jax.lax.rem(metadata_ref.p_id_to_r_base[p_id, 0], align)
    for prefix in range(align):

        @pl.when(r_prefix == prefix)
        def _unpack(prefix=prefix):
            load_rows = pl.cdiv(prefix + aligned_chunk, align) * align
            rows = qkv_slot_ref[0, :load_rows, :]
            dense_ref[8 : 8 + aligned_chunk, :] = rows[
                prefix : prefix + aligned_chunk, :
            ].astype(jnp.float32)

    real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_in_slot_ref=conv_state_in_slot_ref,
        recurrent_in_slot_ref=recurrent_in_slot_ref,
        carry_conv_scratch_ref=carry_conv_scratch_ref,
        carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
        cfg=cfg,
    )
    for k in range(kprev):
        dense_ref[8 - kprev + k : 9 - kprev + k, :] = prev_conv[0, k]

    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    conv_bias = None
    if weights_ref.conv.bias is not None:
        conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

    # x row `base + t` is row t of the compact path's concat(prev_conv, chunk).
    x = dense_ref[0 : 8 + chunk, :]
    base = 8 - kprev
    y = jnp.zeros((chunk, dim), jnp.float32)
    for k in range(cfg.kernel_size):
        y += x[base + k : base + k + chunk] * conv_weight[k]
    if conv_bias is not None:
        y += conv_bias.reshape(1, dim)

    # Conv state: the compact path's checkpoint (window 1), the
    # prev_kernel_size rows ending at the tile's last real token.
    rs = real_sizes[0]
    state = x[base + chunk : base + chunk + kprev]
    for c in range(1, chunk):
        state = jnp.where(c == rs, x[base + c : base + c + kprev], state)
    state3 = state.reshape(kprev, 1, dim)
    if cfg.state_plan is None:
        conv_state_out_slot_ref[0, 0] = state3
    else:
        pl.when(metadata_ref.p_id_is_last_tile[p_id, 0])(
            functools.partial(
                vmem_ldst.store_state_region,
                conv_state_out_slot_ref.at[0, 0],
                cfg.state_plan.conv,
                state3,
            )
        )
    carry_conv_scratch_ref[0] = state3

    act = jax.nn.silu(y)

    padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
    a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

    # Heads are lane ranges of the activated rows, q | k | v as
    # load_activation_as_large reads them.
    dk, dv = cfg.kq_head_dim, cfg.v_head_dim
    k_off = cfg.num_kq_heads * dk
    v_off = 2 * k_off

    def heads(offset, num, width):
        return jnp.stack(
            [act[:, offset + h * width : offset + (h + 1) * width] for h in range(num)],
            axis=0,
        )[None]

    q_large = heads(0, cfg.num_kq_heads, dk)
    k_large = heads(k_off, cfg.num_kq_heads, dk)
    v_large = heads(v_off, cfg.num_v_heads, dv)
    b_large = jnp.expand_dims(vmem_ldst.load_compact_to_large(b_slot_ref), axis=1)
    a_large = jnp.expand_dims(vmem_ldst.load_compact_to_large(a_slot_ref), axis=1)

    out, new_recurrent_state = compute_gdn.chunked_gdn(
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

    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    if cfg.state_plan is None:
        recurrent_out_slot_ref[...] = new_recurrent_state.astype(
            recurrent_out_slot_ref.dtype
        )
    else:
        pl.when(metadata_ref.p_id_is_last_tile[p_id, 0])(
            functools.partial(
                vmem_ldst.store_state_region,
                recurrent_out_slot_ref.at[0, 0],
                cfg.state_plan.recurrent,
                new_recurrent_state[0, 0],
            )
        )
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
    act_scratch_ref: jax.Array | None = None,
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

    scratches = (
        metadata_ref,
        weights_ref,
        carry_conv_scratch_ref,
        carry_recurrent_scratch_ref,
    )
    if act_scratch_ref is not None:
        scratches += (act_scratch_ref,)

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
            scratches=scratches,
            allocations=allocations,
        )

    _run()


def bf16_qkv_2d_eligible(
    *,
    enabled: bool,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    decode_capacity: int,
    mixed_tile_size: int,
    num_lanes: int,
    is_kda: bool = False,
    batched_only: bool = False,
) -> bool:
    """Whether a raw qkv streams into the PER_SEQ kernel as 2D bf16 rows.

    The layout constraints follow the Qwen3.5 branch's bf16_qkv_mode=2: a bf16
    [tokens, dim] input with dim a multiple of 2 * num_lanes, at least one
    16-row tile of tokens and a multiple-of-8 chunk. On top of that, tokens
    must exceed `decode_capacity` (num_seqs * verify window): the BATCHED
    call keeps its f32 rows, cast from that leading slice only, so a bucket
    no larger than it would still cast every row and gains nothing. KDA and
    batched_only calls (no PER_SEQ kernel) keep the f32 path.

    Args:
        enabled: The caller's opt-in (VLLM_TPU_GDN_PREFILL_BF16_QKV_2D).
        shape: Shape of the raw qkv.
        dtype: Dtype of the raw qkv.
        decode_capacity: Most rows the BATCHED segment can address.
        mixed_tile_size: PER_SEQ chunk, already capped at the token count.
        num_lanes: TPU lanes per vreg.
        is_kda: Whether the call runs the KDA gate.
        batched_only: Whether the call skips the PER_SEQ kernel.
    """
    if not enabled or len(shape) != 2 or jnp.dtype(dtype) != jnp.bfloat16:
        return False
    tokens, dim = shape
    return (
        not is_kda
        and not batched_only
        and tokens > decode_capacity
        and tokens >= config.QKV_2D_ROW_ALIGN
        and dim % (2 * num_lanes) == 0
        and mixed_tile_size % 8 == 0
    )


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
        "bf16_qkv_2d",
        "dense_conv",
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
    bf16_qkv_2d: bool | None = None,
    dense_conv: bool | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array] | tuple[jax.Array, jax.Array]:
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
            Ignored when TPU_ENABLE_GDN_DYNAMIC_TILING is set.
        mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
            sequences. Ignored when TPU_ENABLE_GDN_DYNAMIC_TILING is set.
        bf16_qkv_2d: Let a raw bf16 [tokens, dim] qkv reach the PER_SEQ kernel
            as 2D bf16 rows, DMAed in 16-row aligned tiles and cast to f32 in
            VMEM, instead of through an f32 copy in HBM. The BATCHED kernel
            then reads f32 rows cast from the leading num_seqs * window rows
            only. Taken only where `bf16_qkv_2d_eligible` holds; otherwise,
            and for a prepared Conv1DInput, qkv is cast to f32 as before.
            None reads VLLM_TPU_GDN_PREFILL_BF16_QKV_2D.

    Returns:
        (new_conv_state, new_recurrent_state): Updated convolution state cache and
            recurrent state cache tensors. With a state source: the updated
            source (aliased in place) instead of the state pair.
        out: Fused output tensor.
    """
    # QKVColumns: the conv input is the first `cols` columns of a wider bf16
    # projection. The 2D PER_SEQ kernel DMAs those columns in place; every
    # other path slices them out below.
    qkv_cols = None
    if isinstance(qkv, QKVColumns):
        qkv_cols = qkv.cols
        qkv = qkv.values
        if qkv.ndim != 2 or not 0 < qkv_cols <= qkv.shape[1]:
            raise ValueError("QKVColumns requires [tokens, width >= cols]")
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

    # 2D bf16 rows skip the HBM f32 cast; see `bf16_qkv_2d_eligible`.
    if bf16_qkv_2d is None:
        bf16_qkv_2d = envs.VLLM_TPU_GDN_PREFILL_BF16_QKV_2D
    # Token-dense conv of the PER_SEQ tile over those rows; see `_dense_inner_kernel`.
    if dense_conv is None:
        dense_conv = envs.VLLM_TPU_GDN_PREFILL_DENSE_CONV
    decode_capacity = state_indices.size * (num_spec_tokens + 1)
    qkv_2d = not prepared_qkv and bf16_qkv_2d_eligible(
        enabled=bf16_qkv_2d,
        shape=(qkv.shape[0], qkv_cols or qkv.shape[1]),
        dtype=qkv.dtype,
        decode_capacity=decode_capacity,
        mixed_tile_size=min(mixed_tile_size, qkv.shape[0]),
        num_lanes=pltpu.get_tpu_info().num_lanes,
        is_kda=attention_mode == config.AttentionMode.KDA,
        batched_only=batched_only,
    )

    if qkv_cols is not None and not qkv_2d:
        qkv = qkv[:, :qkv_cols]
        qkv_cols = None
    if not prepared_qkv and not qkv_2d:
        qkv = qkv.astype(jnp.float32)
    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)

    # Step 1: Validate inputs.
    is_kda = attention_mode == config.AttentionMode.KDA
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
    batch_size, dim = qkv.shape[0], (qkv_cols or qkv.shape[-1])
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
    assert a.dtype == b.dtype == act_in_dtype

    num_lanes = pltpu.get_tpu_info().num_lanes
    # 2D bf16 rows DMA whole 16-row tiles, so pad the tokens to one.
    packing = config.QKV_2D_ROW_ALIGN if qkv_2d else 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    decode_tile_size = min(decode_tile_size, batch_size)
    mixed_tile_size = min(mixed_tile_size, batch_size)
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
            qkv_2d=qkv_2d,
        )
        if qkv_2d and mixed_tile_size % 8:
            # The VMEM fit can cap the chunk below a multiple of 8, which 2D
            # rows do not take; the 16-padded tokens still suit f32 rows.
            qkv_2d = False
            qkv = qkv[:, :dim].astype(jnp.float32)
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
    qkv_padding = ((0, batch_padding_size),) + ((0, 0),) * (qkv.ndim - 1)
    qkv = jnp.pad(qkv, qkv_padding)
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, gate_padding_size)))

    if not qkv_2d:
        qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = b.reshape(padded_batch_size, 1, -1)
    a = a.reshape(padded_batch_size, 1, -1)

    # Operand of the BATCHED kernel, which keeps f32 compact rows. Its rows
    # are decode/verify windows of at most `num_spec_tokens + 1` tokens for
    # each of the first distribution[0] sequences (query_start_loc[0] == 0),
    # so they all lie in the leading `decode_capacity` rows: cast only those.
    batched_qkv = qkv
    if qkv_2d and not prefill_only:
        batched_qkv = qkv[:decode_capacity, :dim].astype(jnp.float32)
        batched_qkv = batched_qkv.reshape(decode_capacity, 1, dim)

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

    def call_kernel(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array | None,
        in_act: jax.Array | None,
        mode: config.GDNMode,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        # With a state source, it rides the conv-state slot and the dense
        # recurrent-state input is absent.
        if mode == config.GDNMode.PER_SEQ:
            tile_size = mixed_tile_size
            # Prefill / mixed sequences keep a single state checkpoint.
            window_size = 1
            kernel_qkv = qkv
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1
            kernel_qkv = batched_qkv

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
            dtypes=config.Dtypes(
                # bf16 only for 2D rows into PER_SEQ (GDNConfig.qkv_2d).
                act_in=jnp.dtype(kernel_qkv.dtype),
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
        )
        if (
            dense_conv
            and cfg.qkv_2d
            and mode == config.GDNMode.PER_SEQ
            and not cfg.use_recurrent
            and not cfg.is_kda
            and cfg.seq_tile_size == 1
            and cfg.window_size == 1
            and cfg.chunk_size % 8 == 0
        ):
            cfg = dataclasses.replace(cfg, dense_conv=True)

        # Step 6: Metadata preprocessing. Will be executed multiple times per-layer
        # but will be CSEed by compiler.
        if mode != config.GDNMode.PER_SEQ:
            metadata_obj = metadata.compute_batched_seq_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                read_offsets=read_offsets,
                end_seq=distribution[0],
                ckpt_indices=ckpt_indices,
            )
        else:
            metadata_obj = metadata.compute_per_seq_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                read_offsets=read_offsets,
                start_seq=distribution[0],
                end_seq=distribution[-1],
                ckpt_indices=ckpt_indices,
            )

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

        if in_act is None and zero_initialize_out:
            in_act = jnp.zeros_like(out_shape)
        if in_act is not None:
            out_shape = in_act
            in_out_spec = hbm_spec
            input_output_aliases[len(metadata_obj) + 3 + num_state_inputs] = 0

        vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(window_size)

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
            kernel_qkv,
            b,
            a,
            in_conv_state,
            in_recurrent_state,
            in_act,
            weights,
        )

    if batched_only and prefill_only:
        raise ValueError("batched_only and prefill_only are mutually exclusive")

    if pooled:
        out_act, out_source = None, state_source
        if not prefill_only:
            out_act, out_source, _ = call_kernel(
                state_source, None, None, config.GDNMode.BATCHED
            )
        if not batched_only:
            out_act, out_source, _ = call_kernel(
                out_source, None, out_act, config.GDNMode.PER_SEQ
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
