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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import (
    block_doubling,
    compute_conv1d,
    compute_gdn,
    compute_kda,
    config,
    memory_ref,
    metadata,
    tiling,
    vmem_ldst,
)
from vllm_torchtpu.kernels.gdn.v3 import recurrent_state as recurrent_state_lib
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class _TileContext:
    """Tile VMEM refs and post-Conv1D tensors passed to an attention stage."""

    p_id: jax.Array
    real_sizes: jax.Array
    qkv_out_compact: jax.Array
    qkv_slot_ref: jax.Ref
    b_slot_ref: jax.Ref
    a_slot_ref: jax.Ref
    recurrent_in_slot_ref: jax.Ref
    out_slot_ref: jax.Ref
    recurrent_out_slot_ref: jax.Ref
    metadata_ref: memory_ref.MetadataRef
    carry_recurrent_scratch_ref: jax.Ref | None
    qkv_pad_scratch_ref: jax.Ref | None
    a_log: jax.Array
    dt_bias: jax.Array
    cfg: config.GDNConfig


def _should_flush_per_seq_state(ctx: _TileContext) -> jax.Array:
    """Returns whether a single-sequence tile must flush state to HBM."""
    is_last_tile = ctx.metadata_ref.p_id_is_last_tile[ctx.p_id, 0]
    r_size = ctx.metadata_ref.p_id_to_r_size[ctx.p_id, 0]
    should_flush = jnp.logical_and(is_last_tile, r_size > 0)
    if ctx.metadata_ref.s_idx_to_prefill_ckpt_row is not None:
        should_flush = jnp.logical_or(
            should_flush, ctx.metadata_ref.is_prefill_ckpt_tile(ctx.p_id, 0)
        )
    return should_flush


def _run_streamed_recurrent_gdn(ctx: _TileContext) -> None:
    """Runs recurrent GDN via VMEM callbacks (head pairs on 1-token decode)."""
    cfg = ctx.cfg
    q_compact, k_compact, v_compact, b_compact, a_compact = (
        vmem_ldst.load_activation_as_compact(
            qkv_vreg=ctx.qkv_out_compact,
            qkv_vmem_ref=ctx.qkv_slot_ref,
            b_vmem_ref=ctx.b_slot_ref,
            a_vmem_ref=ctx.a_slot_ref,
            cfgs=cfg,
        )
    )
    seq_s_idx = [
        ctx.metadata_ref.p_id_to_s_idx[ctx.p_id, s] for s in range(cfg.seq_tile_size)
    ]
    flush_cond = (
        _should_flush_per_seq_state(ctx)
        if cfg.mode == config.GDNMode.PER_SEQ and cfg.seq_tile_size == 1
        else None
    )
    seq_is_first_tile = (
        [
            ctx.metadata_ref.p_id_is_first_tile[ctx.p_id, s]
            for s in range(cfg.seq_tile_size)
        ]
        if ctx.carry_recurrent_scratch_ref is not None
        else None
    )
    # One-byte FP8 state read directly from HBM on 1-token decode updates in
    # bfloat16.
    fp8_from_hbm = (
        cfg.state_format == config.RecurrentStateFormat.FP8
        and ctx.carry_recurrent_scratch_ref is None
        and cfg.chunk_size == 1
    )
    state_stream = recurrent_state_lib.RecurrentStateStream(
        ctx.recurrent_in_slot_ref,
        ctx.recurrent_out_slot_ref,
        fmt=cfg.state_format,
        n_v=cfg.num_v_heads,
        d_k=cfg.kq_head_dim,
        d_v=cfg.v_head_dim,
        window_size=cfg.window_size,
        seq_has_init=[
            ctx.metadata_ref.s_idx_has_initial_state[s_idx] for s_idx in seq_s_idx
        ],
        carry_ref=ctx.carry_recurrent_scratch_ref,
        seq_is_first_tile=seq_is_first_tile,
        flush_cond=flush_cond,
        load_dtype=jnp.bfloat16 if fp8_from_hbm else jnp.float32,
    )

    out, _ = compute_gdn.recurrent_gdn(
        real_sizes=ctx.real_sizes,
        q_compact=q_compact,
        k_compact=k_compact,
        v_compact=v_compact,
        b_compact=b_compact,
        a_compact=a_compact,
        state_prev=None,
        a_log=ctx.a_log,
        dt_bias=ctx.dt_bias,
        cfg=cfg,
        load_head_fn=state_stream.load_head,
        store_ckpt_head_fn=state_stream.store_ckpt_head,
    )
    ctx.out_slot_ref[...] = out.astype(ctx.out_slot_ref.dtype)


def _run_bulk_attention(ctx: _TileContext) -> None:
    """Runs full-state attention (KDA or sequential chunked GDN)."""
    cfg = ctx.cfg
    assert not recurrent_state_lib.is_scaled(cfg.state_format)
    prev_recurrent = vmem_ldst.load_and_select_recurrent_state(
        metadata_ref=ctx.metadata_ref,
        p_id=ctx.p_id,
        recurrent_in_slot_ref=ctx.recurrent_in_slot_ref,
        carry_recurrent_scratch_ref=ctx.carry_recurrent_scratch_ref,
        cfg=cfg,
    )
    # Non-KDA recurrent GDN is dispatched to _run_streamed_recurrent_gdn.
    if cfg.use_recurrent:
        assert cfg.is_kda, (
            "Non-KDA recurrent GDN must route to _run_streamed_recurrent_gdn."
        )
        q_compact, k_compact, v_compact, b_compact, a_compact = (
            vmem_ldst.load_activation_as_compact(
                qkv_vreg=ctx.qkv_out_compact,
                qkv_vmem_ref=ctx.qkv_slot_ref,
                b_vmem_ref=ctx.b_slot_ref,
                a_vmem_ref=ctx.a_slot_ref,
                cfgs=cfg,
            )
        )
        out, new_recurrent_state = compute_kda.recurrent_kda(
            q_compact=q_compact,
            k_compact=k_compact,
            v_compact=v_compact,
            b_compact=b_compact,
            a_compact=a_compact,
            state_prev=prev_recurrent,
            a_log=ctx.a_log,
            dt_bias=ctx.dt_bias,
            cfg=cfg,
            real_sizes=ctx.real_sizes,
        )
    else:
        chunked_fn = compute_kda.chunked_kda if cfg.is_kda else compute_gdn.chunked_gdn
        q_large, k_large, v_large, b_large, a_large = (
            vmem_ldst.load_activation_as_large(
                qkv_vreg=ctx.qkv_out_compact,
                qkv_vmem_ref=ctx.qkv_slot_ref,
                b_vmem_ref=ctx.b_slot_ref,
                a_vmem_ref=ctx.a_slot_ref,
                cfgs=cfg,
                qkv_pad_vmem_ref=ctx.qkv_pad_scratch_ref,
            )
        )
        out, new_recurrent_state = chunked_fn(
            q_large=q_large,
            k_large=k_large,
            v_large=v_large,
            b_large=b_large,
            a_large=a_large,
            state_prev=prev_recurrent,
            a_log=ctx.a_log,
            dt_bias=ctx.dt_bias,
            cfg=cfg,
            real_sizes=ctx.real_sizes,
        )

    ctx.out_slot_ref[...] = out.astype(ctx.out_slot_ref.dtype)
    vmem_ldst.store_recurrent_state(
        recurrent_out_slot_ref=ctx.recurrent_out_slot_ref,
        carry_recurrent_scratch_ref=ctx.carry_recurrent_scratch_ref,
        new_recurrent_state=new_recurrent_state,
        cfg=cfg,
    )


def _run_block_doubling_gdn(ctx: _TileContext) -> None:
    """Runs block-doubling GDN prefill, streaming Q/K/V, state, and output."""
    cfg = ctx.cfg
    num_lanes = pltpu.get_tpu_info().num_lanes
    state_carry_ref = ctx.carry_recurrent_scratch_ref
    assert state_carry_ref is not None and cfg.seq_tile_size == 1

    qkv_seq_ref = vmem_ldst.stage_qkv_for_strided_load(
        ctx.qkv_out_compact,
        ctx.qkv_slot_ref,
        cfg,
        ctx.qkv_pad_scratch_ref,
    ).at[0]

    def _load_head(kind: vmem_ldst.QKVKind, head: int) -> jax.Array:
        return vmem_ldst.load_qkv_head(qkv_seq_ref, kind, head, cfg, num_lanes)

    def _load_raw_kq(
        kqs: list[int],
    ) -> tuple[dict[int, jax.Array], dict[int, jax.Array]]:
        return (
            {kq: _load_head(vmem_ldst.QKVKind.Q, kq) for kq in kqs},
            {kq: _load_head(vmem_ldst.QKVKind.K, kq) for kq in kqs},
        )

    def _load_raw_v(vs: list[int]) -> dict[int, jax.Array]:
        return {v: _load_head(vmem_ldst.QKVKind.V, v) for v in vs}

    def _load_state_v(v: int) -> jax.Array:
        return state_carry_ref[0, v]

    def _store_state_v(v: int, val: jax.Array) -> None:
        state_carry_ref[0, v] = val

    out_step = block_doubling.out_stream_step(cfg, ctx.out_slot_ref.dtype.itemsize)

    def _store_out_slice(v0: int, heads: list[jax.Array]) -> None:
        blk = jnp.stack(heads, axis=0).swapaxes(0, 1)
        ctx.out_slot_ref[0, :, v0 : v0 + out_step, :] = blk.astype(
            ctx.out_slot_ref.dtype
        )

    b_large, a_large = vmem_ldst.load_ba_as_large(ctx.b_slot_ref, ctx.a_slot_ref)
    out_seq, _ = block_doubling.run_chunk_streamed(
        load_raw_kq=_load_raw_kq,
        load_raw_v=_load_raw_v,
        cfg=cfg,
        real_sizes=ctx.real_sizes,
        b_lanes=b_large[0],
        a_lanes=a_large[0],
        a_log=ctx.a_log,
        dt_bias=ctx.dt_bias,
        load_state_v=_load_state_v,
        store_state_v=_store_state_v,
        store_out_slice=_store_out_slice if out_step else None,
        out_step=out_step,
        out_dtype=ctx.out_slot_ref.dtype if out_step else None,
    )
    if out_seq is not None:
        ctx.out_slot_ref[...] = out_seq.swapaxes(0, 1)[jnp.newaxis].astype(
            ctx.out_slot_ref.dtype
        )

    @pl.when(_should_flush_per_seq_state(ctx))
    def _flush_recurrent_carry() -> None:
        recurrent_state_lib.pack_carry_to_slot(
            state_carry_ref,
            ctx.recurrent_out_slot_ref,
            n_v=cfg.num_v_heads,
            d_k=cfg.kq_head_dim,
            d_v=cfg.v_head_dim,
            fmt=cfg.state_format,
        )


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
    qkv_pad_scratch_ref: jax.Ref | None = None,
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
        qkv_pad_scratch_ref: Optional VMEM scratch reference holding this tile's
            Q/K/V widened to an odd number of lane-width columns
            (`cfg.qkv_pad_scratch_shape`) so strided per-head loads hit distinct
            VMEM banks instead of colliding on `qkv_slot_ref`'s even HBM width.
            Falls back to `qkv_slot_ref` when `None`.
        cfg: GDN configuration object.
        p_id: Optional explicit tile index. Communication-fused outer kernels
            pass their global compute-tile index because their surrounding
            pipeline owns the Pallas program id.
    """

    if p_id is None:
        p_id = pl.program_id(0)

    # Seed block-doubling carry state before Conv1D so the is_first_tile branch
    # does not split Conv1D from the GDN recurrence.
    if cfg.use_block_doubling:
        assert carry_recurrent_scratch_ref is not None and cfg.seq_tile_size == 1
        s_idx = metadata_ref.p_id_to_s_idx[p_id, 0]
        is_first_tile = metadata_ref.p_id_is_first_tile[p_id, 0]

        @pl.when(is_first_tile)
        def _init_recurrent_carry() -> None:
            recurrent_state_lib.unpack_slot_to_carry(
                recurrent_in_slot_ref,
                carry_recurrent_scratch_ref,
                metadata_ref.s_idx_has_initial_state[s_idx],
                n_v=cfg.num_v_heads,
                d_k=cfg.kq_head_dim,
                fmt=cfg.state_format,
            )

    # Prepare states.
    real_sizes, prev_conv = vmem_ldst.load_and_select_conv_state(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_in_slot_ref=conv_state_in_slot_ref,
        carry_conv_scratch_ref=carry_conv_scratch_ref,
        cfg=cfg,
    )

    # Step 1: Conv1D.
    # NOTE: Conv1D requires performing sliding window where inputs are slided
    # across rows. If typical 2D layout was used, multiple rows are stored in a
    # single register which necessitate costly shuffling for every sliding.
    # Therefore, it is extremely important to leverage compact layout that
    # ensures 1 register only stores data from 1 row.
    qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
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

    conv_state_out_slot_ref[...] = new_conv_state.astype(conv_state_out_slot_ref.dtype)
    if carry_conv_scratch_ref is not None:
        # The next tile resumes from the state after this tile's last token,
        # which is the final checkpoint.
        carry_conv_scratch_ref[...] = new_conv_state[:, -1]

    # Apply activation function.
    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    # Step 2: GDN / KDA.
    if cfg.is_kda:
        # KDA reshapes these per (head, channel) rather than padding out to
        # a lane multiple over heads.
        a_log = weights_ref.gdn.a_log[...]
        dt_bias = weights_ref.gdn.dt_bias[...]
    else:
        padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
        a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
        dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

    ctx = _TileContext(
        p_id=p_id,
        real_sizes=real_sizes,
        qkv_out_compact=qkv_out_compact,
        qkv_slot_ref=qkv_slot_ref,
        b_slot_ref=b_slot_ref,
        a_slot_ref=a_slot_ref,
        recurrent_in_slot_ref=recurrent_in_slot_ref,
        out_slot_ref=out_slot_ref,
        recurrent_out_slot_ref=recurrent_out_slot_ref,
        metadata_ref=metadata_ref,
        carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
        qkv_pad_scratch_ref=qkv_pad_scratch_ref,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
    )
    if cfg.use_recurrent and not cfg.is_kda:
        _run_streamed_recurrent_gdn(ctx)
    elif cfg.use_block_doubling:
        _run_block_doubling_gdn(ctx)
    else:
        _run_bulk_attention(ctx)


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    states_ref: jax.Array | tuple[jax.Array, jax.Array],
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    states_out_ref: jax.Array | tuple[jax.Array, jax.Array],
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    qkv_pad_scratch_ref: jax.Array | None = None,
    *,
    cfg: config.GDNConfig,
):
    """Setup memory allocations and emit pipeline for running inner_kernel."""
    if cfg.is_shared:
        # One pool row holds both regions; they are told apart by their
        # granule offset, which `create_allocs` derives from this ref.
        assert not isinstance(states_ref, tuple)
        assert not isinstance(states_out_ref, tuple)
        conv_state_ref = recurrent_state_ref = states_ref
        conv_state_out_ref = recurrent_state_out_ref = states_out_ref
    else:
        assert len(states_ref) == 2
        assert len(states_out_ref) == 2
        conv_state_ref, recurrent_state_ref = states_ref
        conv_state_out_ref, recurrent_state_out_ref = states_out_ref

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
    body = functools.partial(inner_kernel, cfg=cfg)
    scratches = (
        metadata_ref,
        weights_ref,
        carry_conv_scratch_ref,
        carry_recurrent_scratch_ref,
        qkv_pad_scratch_ref,
    )
    state_allocs = (conv_in_alloc, recurrent_in_alloc)
    out_allocs = (out_alloc, conv_out_alloc, recurrent_out_alloc)
    state_out_args = (conv_state_out_ref, recurrent_state_out_ref)

    pipeline_func = pltpu.emit_pipeline(
        body=body,
        grid=(num_tiles,),
        in_specs=(
            qkv_alloc.spec,
            b_alloc.spec,
            a_alloc.spec,
            *(alloc.spec for alloc in state_allocs),
        ),
        out_specs=tuple(alloc.spec for alloc in out_allocs),
    )

    @pl.with_scoped(
        allocations=(
            qkv_alloc,
            b_alloc,
            a_alloc,
            *state_allocs,
            *out_allocs,
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
            *state_out_args,
            scratches=scratches,
            allocations=allocations,
        )

    _run()


def _resolve_gdn_prefill_mode(
    gdn_prefill_mode: config.GDNPrefillMode | str | None,
    *,
    attention_mode: config.AttentionMode,
    state_fmt: config.RecurrentStateFormat,
    d_k: int,
    d_v: int,
) -> config.GDNPrefillMode:
    """Resolves and validates `gdn_prefill_mode` against model and TPU limits."""
    is_kda = attention_mode == config.AttentionMode.KDA
    is_scaled = recurrent_state_lib.is_scaled(state_fmt)
    if gdn_prefill_mode is not None:
        mode = config.GDNPrefillMode(gdn_prefill_mode)
        if is_kda and mode != config.GDNPrefillMode.SEQUENTIAL:
            raise ValueError(
                f"gdn_prefill_mode={mode.value!r} only applies to GDN; "
                f"got attention_mode={attention_mode}."
            )
        if is_scaled and mode == config.GDNPrefillMode.SEQUENTIAL:
            raise NotImplementedError(
                f"recurrent_state_format={state_fmt.value!r} requires a "
                "block-doubling gdn_prefill_mode ('block_doubling_bf16' or "
                "'block_doubling_fp8')."
            )
        resolved: config.GDNPrefillMode | None = mode
    else:
        resolved = config.GDNPrefillMode.BLOCK_DOUBLING_BF16 if is_scaled else None

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    if resolved is None:
        if not is_kda and d_k % num_lanes == 0 and d_v % num_lanes == 0:
            return config.GDNPrefillMode.BLOCK_DOUBLING_BF16
        return config.GDNPrefillMode.SEQUENTIAL
    if resolved != config.GDNPrefillMode.SEQUENTIAL and (
        d_k % num_lanes or d_v % num_lanes
    ):
        raise ValueError(
            f"gdn_prefill_mode={resolved.value!r} requires d_k and d_v to be "
            f"multiples of {num_lanes}; got d_k={d_k}, d_v={d_v}."
        )
    mxu_operand_dtype = resolved.mxu_operand_dtype
    if mxu_operand_dtype is not None and not tpu_info.is_matmul_supported(
        mxu_operand_dtype, mxu_operand_dtype
    ):
        raise NotImplementedError(
            f"The MXU of TPU {tpu_info.chip_version} does not take "
            f"{mxu_operand_dtype.name} operands."
        )
    return resolved


def _resolve_state_format(
    states: jax.Array | tuple[jax.Array, jax.Array],
    *,
    recurrent_dtype: jnp.dtype | None,
    recurrent_state_format: config.RecurrentStateFormat | str | None,
    attention_mode: config.AttentionMode,
    n_v: int,
    d_k: int,
    d_v: int,
) -> tuple[config.RecurrentStateFormat, jnp.dtype]:
    """Resolves and validates recurrent state format and tensor dtype/shape."""
    if (
        recurrent_state_format is None
        and recurrent_dtype is not None
        and jnp.dtype(recurrent_dtype) == jnp.uint32
    ):
        raise ValueError(
            "Received uint32 recurrent_state without recurrent_state_format; "
            "pass recurrent_state_format='fp8' to use packed FP8 recurrent state."
        )
    state_fmt = recurrent_state_lib.resolve_format(
        recurrent_state_format, recurrent_dtype
    )
    is_scaled = recurrent_state_lib.is_scaled(state_fmt)
    if is_scaled:
        if isinstance(states, jax.Array):
            raise NotImplementedError(
                "Packed scaled recurrent state is not yet supported with "
                "unified-pool shared states."
            )
        expected_rows = recurrent_state_lib.packed_rows(state_fmt, n_v, d_k)
        rec = states[1]
        if rec.dtype != jnp.uint32 or rec.shape[1:] != (expected_rows, d_v):
            raise ValueError(
                "Expected uint32 recurrent_state of shape [num_blocks, "
                f"{expected_rows}, {d_v}] for {state_fmt.value}; got "
                f"{rec.dtype}[{rec.shape}]."
            )
        if recurrent_dtype is not None and jnp.dtype(recurrent_dtype) != jnp.uint32:
            raise ValueError(
                f"recurrent_state_format={state_fmt.value!r} requires "
                f"recurrent_dtype=uint32; got {jnp.dtype(recurrent_dtype).name}."
            )
        if attention_mode == config.AttentionMode.KDA:
            raise NotImplementedError(
                f"recurrent_state_format={state_fmt.value!r} is not supported for KDA."
            )
        return state_fmt, jnp.dtype(jnp.uint32)

    fmt_dtype = jnp.dtype(
        jnp.bfloat16 if state_fmt == config.RecurrentStateFormat.BF16 else jnp.float32
    )
    if recurrent_dtype is not None and jnp.dtype(recurrent_dtype) != fmt_dtype:
        raise ValueError(
            f"recurrent_state_format={state_fmt.value!r} requires "
            f"recurrent_dtype={fmt_dtype.name}; got "
            f"{jnp.dtype(recurrent_dtype).name}."
        )
    return state_fmt, fmt_dtype


@jax.jit(
    donate_argnames=("states",),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "num_spec_tokens",
        "decode_tile_size",
        "mixed_tile_size",
        "batched_only",
        "prefill_only",
        "zero_initialize_out",
        "compute_precision",
        "gdn_prefill_mode",
        "recurrent_dtype",
        "recurrent_state_format",
        "conv_dtype",
        "attention_mode",
        "gate_lower_bound",
        "kernel_act_dtype",
        "kernel_conv_weight_dtype",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
    b: jax.Array,  # [batch_size, n_v]
    a: jax.Array,  # [batch_size, n_v]; [batch_size, n_v * d_k] for KDA
    states: jax.Array | tuple[jax.Array, jax.Array],
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
    prefill_ckpt_offsets: jax.Array | None = None,  # [num_seqs]
    prefill_ckpt_indices: jax.Array | None = None,  # [num_seqs]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    gate_lower_bound: float | None = None,
    recurrent_dtype: jnp.dtype | None = None,
    recurrent_state_format: config.RecurrentStateFormat | str | None = None,
    conv_dtype: jnp.dtype | None = None,
    num_spec_tokens: int = 0,
    batched_only: bool = False,
    prefill_only: bool = False,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    gdn_prefill_mode: config.GDNPrefillMode | str | None = None,
    kernel_act_dtype: jnp.dtype = jnp.float32.dtype,
    kernel_conv_weight_dtype: jnp.dtype = jnp.float32.dtype,
    # TODO(kyuyeunk): Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
) -> tuple[jax.Array | tuple[jax.Array, jax.Array], jax.Array]:
    """Perform conv1d and gdn in a single fused kernel.

    Args:
        qkv: Mixed query, key, value input tensor of shape [batch_size, dim_size],
            where `dim_size = n_kq * d_k * 2 + n_v * d_v`.
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
        prefill_ckpt_offsets: Optional [num_seqs] int32. Query token offset
            of each sequence's internal prefill checkpoint (0 for none). When
            0 < offset < query_len and prefill_ckpt_indices is non-negative,
            the state after the first offset query tokens is also written,
            encoded like the final state, to block prefill_ckpt_indices[s].
            Must be passed together with prefill_ckpt_indices.
        prefill_ckpt_indices: Optional [num_seqs] int32. Destination state
            block of each sequence's prefill checkpoint (-1 for none).
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
        num_spec_tokens: Number of speculative draft tokens (0 gives the
            plain 1-token-per-sequence decode path).
        batched_only: Run only the leading decode/verify segment.
        prefill_only: Run only PER_SEQ, leaving decode/verify checkpoints
            untouched. Mutually exclusive with batched_only.
        zero_initialize_out: Whether to zero-initialize the output buffer before
            executing non-batched sequences.
        compute_precision: Working dtype of the GDN / KDA stage. `q`, `k`, `v`,
            `b`, `a`, `a_log` and `dt_bias` are cast to it after the conv1d
            stage; sequential GDN and KDA round intermediates back to it between
            ops, while block-doubling modes use bfloat16/FP8 intermediates. The
            conv1d stage always runs in float32.
        gdn_prefill_mode: Chunked GDN prefill recurrence mode, a
            `GDNPrefillMode`, its string value, or `None`. `None` (the default)
            selects `"block_doubling_bf16"` for GDN when `d_k` and `d_v` are
            multiples of the TPU lane count (covering both Qwen 3.8 and
            Qwen 3.5) or when using a scaled `recurrent_state_format`, and
            `"sequential"` for KDA or non-lane-aligned head dims. `"sequential"`
            opts out to the sequential chunked kernel; `"block_doubling_bf16"`
            and `"block_doubling_fp8"` select Schur block doubling with 16-bit
            or 8-bit (`float8_e4m3fn`) MXU operands for supported chunk sizes
            (`32`, `64`, or `>= 128` divisible by `128`), falling back to
            `"sequential"` on other chunk sizes (GDN only; explicit
            block-doubling modes raise `ValueError` under KDA or
            non-lane-aligned `d_k`/`d_v`, and `NotImplementedError` when FP8
            MXU is unavailable).
        recurrent_state_format: Optional packed storage format for the recurrent
            state (`"f32"`, `"bf16"`, or `"fp8"`). When `None`, inferred from
            `recurrent_dtype`. `"fp8"` expects `uint32`
            `[num_blocks, packed_rows, d_v]` from `recurrent_state.pack_state`
            and requires GDN with separate `(conv, recurrent)` states and
            streamed recurrent decode/verify or block-doubling prefill.
        kernel_act_dtype: Dtype qkv, b and a are cast to before the kernel.
        kernel_conv_weight_dtype: Dtype conv_weight and conv_bias are cast to
            before the kernel.
        decode_tile_size: Tile size along sequence dimension for decode sequences.
            Ignored when TPU_ENABLE_GDN_DYNAMIC_TILING is set.
        mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
            sequences. Ignored when TPU_ENABLE_GDN_DYNAMIC_TILING is set.

    Returns:
        new_states: The updated state cache, aliased in place, with the same
            structure as `states`.
        out: Fused output tensor.
    """
    # TODO(kyuyeunk): Support bf16
    if jnp.dtype(kernel_act_dtype) != jnp.float32:
        raise ValueError(
            f"kernel_act_dtype must be float32; got {jnp.dtype(kernel_act_dtype).name}."
        )

    act_out_dtype = qkv.dtype

    is_shared = isinstance(states, jax.Array)
    if is_shared:
        state_packing = 4 // states.dtype.itemsize
        # Ensure compact layout is used.
        assert states.shape[-2] == state_packing
        assert conv_dtype is not None
        if recurrent_dtype is None and recurrent_state_format is None:
            raise ValueError(
                "Shared state pool requires recurrent_dtype or recurrent_state_format."
            )
    else:
        assert len(states) == 2
        if (
            recurrent_dtype is not None
            and jnp.dtype(recurrent_dtype) != states[1].dtype
        ):
            raise ValueError(
                f"recurrent_dtype={jnp.dtype(recurrent_dtype).name} conflicts "
                f"with recurrent state tensor dtype {states[1].dtype.name}."
            )
        recurrent_dtype = states[1].dtype
        conv_dtype = states[0].dtype
    # Conv1D reads the compact layout, which needs fp32.
    if jnp.dtype(conv_dtype) != jnp.float32:
        raise NotImplementedError(
            f"GDN v3 needs an fp32 conv state; got conv={conv_dtype}."
        )
    state_fmt, recurrent_dtype = _resolve_state_format(
        states,
        recurrent_dtype=recurrent_dtype,
        recurrent_state_format=recurrent_state_format,
        attention_mode=attention_mode,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
    )

    qkv = qkv.astype(kernel_act_dtype)
    b = b.astype(kernel_act_dtype)
    a = a.astype(kernel_act_dtype)

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
    batch_size, dim = qkv.shape
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
        read_offsets = jnp.zeros_like(state_indices)
    assert read_offsets.shape == (num_seqs,)
    if ckpt_indices is not None:
        # One source block per checkpoint (vLLM's ssm_state_indices scheme):
        # checkpoint t of sequence s lives at block ckpt_indices[s, t], so
        # the pool block only ever has to hold a single state.
        assert ckpt_indices.shape == (num_seqs, num_spec_tokens + 1)
    else:
        # Iniitalize the ckpt_indices to default value.
        spec_offset = jnp.arange(num_spec_tokens + 1).reshape(1, -1)
        ckpt_indices = state_indices.reshape(-1, 1) + spec_offset
    assert (prefill_ckpt_offsets is None) == (prefill_ckpt_indices is None), (
        "prefill_ckpt_offsets and prefill_ckpt_indices go together"
    )
    if prefill_ckpt_offsets is not None:
        assert prefill_ckpt_indices is not None
        assert prefill_ckpt_offsets.shape == (num_seqs,), prefill_ckpt_offsets.shape
        assert prefill_ckpt_indices.shape == (num_seqs,), prefill_ckpt_indices.shape
        prefill_ckpt_offsets = prefill_ckpt_offsets.astype(jnp.int32)
        prefill_ckpt_indices = prefill_ckpt_indices.astype(jnp.int32)

    act_in_dtype = qkv.dtype
    assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

    gdn_prefill_mode = _resolve_gdn_prefill_mode(
        gdn_prefill_mode,
        attention_mode=attention_mode,
        state_fmt=state_fmt,
        d_k=d_k,
        d_v=d_v,
    )
    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    decode_tile_size = min(decode_tile_size, batch_size)
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
            conv_state_dim_size=dim if is_shared else states[0].shape[-1],
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=recurrent_dtype,
            num_lanes=num_lanes,
            decode_vmem_limit_bytes=decode_vmem_limit_bytes,
            mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
            window_size=window_size,
            recurrent_state_format=state_fmt,
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
        # Round the static clip down to a power of two so a small
        # non-power-of-two token bucket (e.g. TPU_TOKEN_BUCKET_EXTRA or tiny
        # max_num_batched_tokens) stays divisible by triangular_block_size.
        mixed_tile_size = tiling.clip_mixed_tile_size(mixed_tile_size, batch_size)
        # Each sequence holds 1 input state slot plus `window` output
        # checkpoint slots per buffer in VMEM, at every window size. Shrink
        # the tile so the double-buffered windows fit in roughly half the
        # scoped-VMEM budget.
        window = num_spec_tokens + 1
        num_buffers = config.GDNConfig.__dataclass_fields__["num_buffers"].default
        state_positions = 1 + window
        recurrent_bytes_per_seq = state_positions * recurrent_state_lib.slot_bytes(
            state_fmt, n_v, d_k, d_v
        )
        bytes_per_seq = (
            # Recurrent checkpoints — the dominant term.
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
    qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, gate_padding_size)))

    qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = b.reshape(padded_batch_size, 1, -1)
    a = a.reshape(padded_batch_size, 1, -1)

    # TODO(kyuyeunk): Ensure no reshape is needed and convert to assertion.
    conv_state_shape = None
    recurrent_state_shape = None
    if not is_shared:
        conv_state_shape = states[0].shape
        rec_state = states[1]
        if jnp.dtype(recurrent_dtype) == jnp.bfloat16:
            recurrent_state_shape = rec_state.shape
            rec_state = rec_state.reshape(-1, n_v * (d_k // 2), 2, d_v)
        states = (states[0].reshape(-1, kernel_size - 1, 1, dim), rec_state)

    # Step 3: Weights pre-processing.
    # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
    # loading stage.
    conv_weight = conv_weight.swapaxes(0, 2).astype(kernel_conv_weight_dtype)
    if conv_bias is not None:
        conv_bias = conv_bias.astype(kernel_conv_weight_dtype)

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
        in_states: jax.Array | tuple[jax.Array, jax.Array],
        in_act: jax.Array | None,
        mode: config.GDNMode,
    ) -> tuple[jax.Array, jax.Array | tuple[jax.Array, jax.Array]]:
        if mode == config.GDNMode.PER_SEQ:
            tile_size = mixed_tile_size
            # Prefill / mixed sequences keep a single state checkpoint.
            window_size = 1
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1

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
            gdn_prefill_mode=gdn_prefill_mode,
            recurrent_state_format=state_fmt,
            dtypes=config.Dtypes(
                act_in=act_in_dtype,
                act_out=act_out_dtype,
                compute=compute_precision,
                recurrent_state=recurrent_dtype,
                conv_state=conv_dtype,
            ),
            is_shared=is_shared,
        )

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
                prefill_ckpt_offsets=prefill_ckpt_offsets,
                prefill_ckpt_indices=prefill_ckpt_indices,
            )

        vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(window_size)
        metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)
        state_spec = jax.tree.map(lambda _: hbm_spec, in_states)
        scratch_shapes = cfg.get_scratch_shape_dict()
        qpad_shape = cfg.qkv_pad_scratch_shape(num_lanes)
        if qpad_shape is not None and not tiling.fits_qkv_pad_scratch(
            cfg, qpad_shape, vmem_limit_bytes, num_lanes
        ):
            logger.info_once(
                "Skipping odd-column Q/K/V VMEM scratchpad %s for chunk_size=%d,"
                " num_v_heads=%d (exceeds vmem_limit_bytes=%d).",
                qpad_shape,
                cfg.chunk_size,
                cfg.num_v_heads,
                vmem_limit_bytes,
            )
            qpad_shape = None
        scratch_shapes["qkv_pad_scratch_ref"] = (
            None if qpad_shape is None else pltpu.VMEM(qpad_shape, jnp.float32)
        )

        # Step 7: Handle case where write needs to be done in existing out.
        input_output_aliases = {}
        state_pos = len(metadata_obj) + 3
        # This ensures the logic can support both unified and non-unified pool.
        num_states = len(jax.tree.leaves(in_states))
        for i in range(num_states):
            input_output_aliases[state_pos + i] = 1 + i
        act_pos = state_pos + num_states

        in_out_spec = None
        out_shape = cfg.get_out_shape()
        if in_act is None and zero_initialize_out:
            in_act = jnp.zeros_like(out_shape)
        if in_act is not None:
            out_shape = in_act
            in_out_spec = hbm_spec
            input_output_aliases[act_pos] = 0

        return pl.pallas_call(
            functools.partial(outer_kernel, cfg=cfg),
            out_shape=(out_shape, in_states),
            in_specs=(
                metadata_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                state_spec,
                in_out_spec,
                weights_spec,
            ),
            out_specs=(hbm_spec, state_spec),
            scratch_shapes=scratch_shapes,
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
            in_states,
            in_act,
            weights,
        )

    # The first segment holds verify windows of up to `num_spec_tokens + 1`
    # tokens, or plain 1-token decodes without speculative decoding.
    if batched_only and prefill_only:
        raise ValueError("batched_only and prefill_only are mutually exclusive")

    out_act, out_states = None, states
    if not prefill_only:
        out_act, out_states = call_kernel(states, None, config.GDNMode.BATCHED)
    if not batched_only:
        out_act, out_states = call_kernel(out_states, out_act, config.GDNMode.PER_SEQ)

    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    # TODO(kyuyeunk): Ensure no reshape is needed and convert to assertion.
    if conv_state_shape is not None:
        out_states = (out_states[0].reshape(conv_state_shape), out_states[1])
    if recurrent_state_shape is not None:
        out_states = (out_states[0], out_states[1].reshape(recurrent_state_shape))
    return out_states, out_act
