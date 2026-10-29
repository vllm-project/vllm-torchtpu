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

import contextlib
import functools
import json
import os
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import static_guards
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
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _decode_tile_from_fused_ep_config() -> int:
    """The decode tile size carried by TPU_MOE_FUSED_EP_V3_CONFIG's
    "gdn_decode_tile_size" field (0: none). The serving recipe passes that
    JSON env to the decode server alone and drops unknown env names such as
    TPU_GDN_DECODE_TILE_SIZE, so the JSON is the route that reaches decode
    and never reaches prefill."""
    raw = os.environ.get("TPU_MOE_FUSED_EP_V3_CONFIG", "").strip()
    if not raw:
        return 0
    try:
        value = json.loads(raw).get("gdn_decode_tile_size", 0)
        return int(value or 0)
    except (ValueError, AttributeError, TypeError):
        return 0


def _decode_tile_override() -> int:
    if os.environ.get("TPU_GDN_SPEC_WINDOW_TILE1", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return 1
    raw = os.environ.get("TPU_GDN_DECODE_TILE_SIZE") or os.environ.get(
        "TPU_GDN_DECODE_TILE_OVERRIDE", "0"
    )
    try:
        tile = int(raw)
    except ValueError:
        tile = 0
    if tile <= 0:
        tile = _decode_tile_from_fused_ep_config()
        if tile > 0:
            logger.info_once(
                "GDN decode tile size override %d (from TPU_MOE_FUSED_EP_V3_CONFIG "
                "gdn_decode_tile_size)",
                tile,
            )
    return tile


def _pooled_region_bytes(
    state_source: jax.Array, plan: config.StateSourcePlan, region: config.StateRegion
) -> int:
    """VMEM bytes of one checkpoint's tile of `region` (create_allocs' shape)."""
    itemsize = jnp.dtype(state_source.dtype).itemsize
    if plan.whole_block_dma:
        per_block = 1
        for d in state_source.shape[1:]:
            per_block *= d
        return region.nblocks * per_block * itemsize
    payload = 1
    for d in state_source.shape[2:]:
        payload *= d
    return region.nblocks * region.nrows * payload * itemsize


def _scope(name: str):
    """A profiler scope around a kernel phase, only under TPU_GDN_TRACE_SCOPES."""
    if config.trace_scopes():
        return jax.named_scope(name)
    return contextlib.nullcontext()


def _bf16_qkv_mode() -> int:
    raw = os.environ.get("TPU_GDN_BF16_QKV", "2").strip()
    try:
        return int(raw)
    except ValueError:
        return 2


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size] or bf16
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
    ckpt_stage_ref: jax.Ref | None = None,
    ckpt_sems: jax.Ref | None = None,
    dense_act_scratch_ref: jax.Ref | None = None,
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
        act_scratch_ref: Optional VMEM scratch reference for post-conv1d f32
            activations when qkv_slot_ref is bfloat16.
        ckpt_stage_ref: With `cfg.manual_ckpt_dma`, the two-slot staging
            tile [2, seq_tile_size, window_size, nblocks, *page] the
            recurrent checkpoints are encoded into before the body DMAs
            them to the source; `recurrent_out_slot_ref` is then the source
            itself.
        ckpt_sems: The staging tile's per-slot DMA semaphores.
        dense_act_scratch_ref: With `cfg.dense_conv`, the token-dense f32
            rows [8 + aligned chunk, dim_size] the bf16 qkv tile is unpacked
            into (rows [8 - prev_kernel_size, 8) carry the previous conv rows).
        cfg: GDN configuration object.
        p_id: Optional explicit tile index. Communication-fused outer kernels
            pass their global compute-tile index because their surrounding
            pipeline owns the Pallas program id.
    """

    if p_id is None:
        p_id = pl.program_id(0)

    # 2D bf16 qkv: shift each sequence's rows out of their 16-row aligned DMA
    # window into the f32 scratch. This runs before load_and_select_states so
    # the unpack's bf16 load and f32 cast are dead before the dense path's
    # selected states become live. Ordered after it, the unpack temporaries
    # add to the VMEM peak: 4.0 MiB more for the per_seq b1 c64 kernel at dim
    # 12288 on TPU7x.
    qkv_work_ref = qkv_slot_ref
    dense = cfg.dense_conv
    stub = config.bench_stub()
    if qkv_slot_ref.ndim == 3 and not dense:
        assert act_scratch_ref is not None
        aligned_chunk = pl.cdiv(cfg.chunk_size, 16) * 16
        for idx in range(cfg.seq_tile_size):
            r_base = metadata_ref.p_id_to_r_base[p_id, idx]
            r_prefix = jax.lax.rem(r_base, 16)
            for prefix in range(16):

                @pl.when(r_prefix == prefix)
                def _unpack(idx=idx, prefix=prefix):
                    load_rows = pl.cdiv(prefix + aligned_chunk, 16) * 16
                    dense_qkv = qkv_slot_ref[idx, :load_rows, :]
                    act_scratch_ref[idx, :aligned_chunk, 0, :] = dense_qkv[
                        prefix : prefix + aligned_chunk, :
                    ].astype(jnp.float32)

        qkv_work_ref = (
            act_scratch_ref
            if aligned_chunk == cfg.chunk_size
            else act_scratch_ref.at[:, : cfg.chunk_size]
        )

    # Pooled BATCHED GDN tiles can encode each checkpoint inside the token
    # loop, so neither the incoming nor the outgoing f32 states are stacked.
    manual_dma = None
    pooled_batched = (
        cfg.state_plan is not None
        and cfg.use_recurrent
        and not cfg.is_kda
        and carry_recurrent_scratch_ref is None
    )
    inline_ckpt = (
        config.ckpt_store_inline() or cfg.manual_ckpt_dma
    ) and pooled_batched
    # Grouped tiles decode and encode the recurrent state per head group
    # inside the chain, so the tile never holds a whole f32 state.
    head_group = config.head_group()
    grouped = (
        head_group > 0
        and pooled_batched
        and vmem_ldst.head_rows_supported(cfg)
        and cfg.num_v_heads % head_group == 0
    )

    setup_scope = _scope("gdn_setup")
    setup_scope.__enter__()
    # Prepare states.
    real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_in_slot_ref=conv_state_in_slot_ref,
        recurrent_in_slot_ref=recurrent_in_slot_ref,
        carry_conv_scratch_ref=carry_conv_scratch_ref,
        carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
        cfg=cfg,
        stack_recurrent=not inline_ckpt,
        load_recurrent=not grouped,
    )

    # Prepare conv1d weights.
    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    conv_bias = None
    if weights_ref.conv.bias is not None:
        conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

    dense_act = None
    if dense:
        dense_act = _dense_conv_silu(
            qkv_slot_ref=qkv_slot_ref,
            dense_ref=dense_act_scratch_ref,
            prev_conv=prev_conv,
            real_sizes=real_sizes,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            conv_state_out_slot_ref=conv_state_out_slot_ref,
            carry_conv_scratch_ref=carry_conv_scratch_ref,
            metadata_ref=metadata_ref,
            p_id=p_id,
            cfg=cfg,
        )
        qkv_out_compact = None
    else:
        # Step 1: Conv1D.
        # NOTE: Conv1D requires performing sliding window where inputs are
        # slided across rows. If typical 2D layout was used, multiple rows are
        # stored in a single register which necessitate costly shuffling for
        # every sliding. Therefore, it is extremely important to leverage
        # compact layout that ensures 1 register only stores data from 1 row.
        if qkv_slot_ref.ndim == 3:
            qkv_in_compact = qkv_work_ref[...]
        elif qkv_slot_ref.shape[-2] == 2:
            assert act_scratch_ref is not None
            qkv_raw = qkv_slot_ref[...].astype(jnp.float32)
            qkv_in_compact = jnp.concatenate(
                [qkv_raw[:, :, 0:1, :], qkv_raw[:, :, 1:2, :]], axis=-1
            )
            qkv_work_ref = act_scratch_ref
        else:
            qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
        qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

        conv_scope = _scope("gdn_conv")
        conv_scope.__enter__()
        lazy_state = (
            config.prefill_lazy_conv_state()
            and cfg.mode == config.GDNMode.PER_SEQ
            and not cfg.use_recurrent
            and carry_conv_scratch_ref is not None
            and cfg.window_size == 1
            and cfg.seq_tile_size == 1
        )
        if stub == "conv":
            # Attribution stand-in: no taps, the tail rows are the state.
            qkv_out_compact = qkv_in_compact[:, cfg.prev_kernel_size :]
            new_conv_state = jnp.stack(
                [qkv_in_compact[:, -cfg.prev_kernel_size :]] * cfg.window_size, axis=1
            )
        elif lazy_state:
            qkv_out_compact = compute_conv1d.causal_conv1d_out(
                qkv_in_compact, conv_weight, conv_bias, cfg
            )
            new_conv_state = None
            _lazy_conv_state(
                lhs=qkv_in_compact[0],
                real_sizes=real_sizes,
                conv_state_out_slot_ref=conv_state_out_slot_ref,
                carry_conv_scratch_ref=carry_conv_scratch_ref,
                metadata_ref=metadata_ref,
                p_id=p_id,
                cfg=cfg,
            )
        else:
            qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
                real_sizes=real_sizes,
                lhs=qkv_in_compact,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                cfg=cfg,
            )

    if dense or new_conv_state is None:
        # The conv state was stored by the dense or lazy path.
        pass
    elif cfg.state_plan is None:
        conv_state_out_slot_ref[...] = new_conv_state
    else:
        # External source tiles: encode each window position's checkpoint
        # through the region's typed view; copy_out pushes the raw bytes
        # of the valid checkpoints back to the source.
        first_ckpt = cfg.window_size - config.bench_ckpt_writes(cfg.window_size)
        for idx in range(cfg.seq_tile_size):
            for t in range(first_ckpt, cfg.window_size):
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
    if carry_conv_scratch_ref is not None and not dense and new_conv_state is not None:
        # The next tile resumes from the state after this tile's last token,
        # which is the final checkpoint.
        carry_conv_scratch_ref[...] = new_conv_state[:, -1]

    # Apply activation function.
    if not dense:
        qkv_out_compact = jax.nn.silu(qkv_out_compact)
        conv_scope.__exit__(None, None, None)

    # Step 2: GDN.

    # Prepare gdn weights (pre-padded to aligned_num_v_heads outside pallas_call
    # for non-KDA; fallback pad kept for direct inner_kernel unit tests).
    if cfg.is_kda or weights_ref.gdn.a_log.shape[0] == cfg.aligned_num_v_heads:
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

        recurrent_kwargs = {}
        deferred_dma = cfg.manual_ckpt_dma and config.ckpt_manual_dma_deferred()
        if cfg.manual_ckpt_dma:
            assert inline_ckpt and ckpt_stage_ref is not None
            manual_dma = _ManualCkptDma(
                metadata_ref=metadata_ref,
                p_id=p_id,
                stage_ref=ckpt_stage_ref,
                sems=ckpt_sems,
                source_ref=recurrent_out_slot_ref,
                cfg=cfg,
            )
            # The slot about to be filled was last used two steps ago.
            manual_dma.wait_step(p_id - 2, pl.when(p_id >= 2))
            if deferred_dma:
                # The previous step's checkpoints leave for the source between
                # this step's setup and its token chain.
                manual_dma.start_step(p_id - 1, pl.when(p_id >= 1))
        if inline_ckpt:
            first_ckpt = cfg.window_size - config.bench_ckpt_writes(cfg.window_size)

            def _emit_state(idx: int, t: int, state: jax.Array) -> None:
                if t < first_ckpt:
                    return
                if manual_dma is None:
                    vmem_ldst.store_recurrent_state(
                        recurrent_out_slot_ref.at[idx, t],
                        conv_state_out_slot_ref.at[idx, t],
                        cfg,
                        state,
                    )
                    return
                vmem_ldst.store_recurrent_state(
                    manual_dma.stage.at[idx, t],
                    conv_state_out_slot_ref.at[idx, t],
                    cfg,
                    state,
                )
                if not deferred_dma:
                    manual_dma.start(idx, t)

            recurrent_kwargs["emit_state"] = _emit_state

        setup_scope.__exit__(None, None, None)
        chain_scope = _scope("gdn_chain")
        chain_scope.__enter__()
        if grouped:
            assert prev_recurrent is None
            first_ckpt = cfg.window_size - config.bench_ckpt_writes(cfg.window_size)
            has_initial = [
                metadata_ref.s_idx_has_initial_state[metadata_ref.p_id_to_s_idx[p_id, idx]]
                for idx in range(cfg.seq_tile_size)
            ]

            def _load_heads(idx: int, h0: int, nh: int) -> jax.Array:
                return vmem_ldst.load_recurrent_heads(
                    recurrent_in_slot_ref.at[idx, 0], cfg, h0, nh, has_initial[idx]
                )

            def _emit_heads(idx: int, t: int, h0: int, nh: int, state: jax.Array):
                if t < first_ckpt:
                    return
                dst = (
                    recurrent_out_slot_ref if manual_dma is None else manual_dma.stage
                )
                vmem_ldst.store_recurrent_heads(dst.at[idx, t], cfg, h0, nh, state)

            def _after_seq(idx: int) -> None:
                # Every head group of the sequence's checkpoints is encoded.
                if manual_dma is not None and not deferred_dma:
                    for t in range(first_ckpt, cfg.window_size):
                        manual_dma.start(idx, t)

            out = compute_gdn.recurrent_gdn_grouped(
                real_sizes=real_sizes,
                q_compact=q_compact,
                k_compact=k_compact,
                v_compact=v_compact,
                b_compact=b_compact,
                a_compact=a_compact,
                a_log=a_log,
                dt_bias=dt_bias,
                cfg=cfg,
                group=head_group,
                load_state=_load_heads,
                emit_state=_emit_heads,
                after_seq=_after_seq,
            )
            new_recurrent_state = None
        else:
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
                **recurrent_kwargs,
            )
        chain_scope.__exit__(None, None, None)

    else:
        chunked_fn = compute_kda.chunked_kda if cfg.is_kda else compute_gdn.chunked_gdn
        if dense:
            assert dense_act is not None
            # Heads are lane ranges of the dense rows; the split is free.
            dk, dv = cfg.kq_head_dim, cfg.v_head_dim
            k_off = cfg.num_kq_heads * dk
            v_off = 2 * k_off
            q_large = jnp.stack(
                [dense_act[:, h * dk : (h + 1) * dk] for h in range(cfg.num_kq_heads)],
                axis=0,
            )[None]
            k_large = jnp.stack(
                [
                    dense_act[:, k_off + h * dk : k_off + (h + 1) * dk]
                    for h in range(cfg.num_kq_heads)
                ],
                axis=0,
            )[None]
            v_large = jnp.stack(
                [
                    dense_act[:, v_off + h * dv : v_off + (h + 1) * dv]
                    for h in range(cfg.num_v_heads)
                ],
                axis=0,
            )[None]
            b_large = jnp.expand_dims(vmem_ldst.load_compact_to_large(b_slot_ref), axis=1)
            a_large = jnp.expand_dims(vmem_ldst.load_compact_to_large(a_slot_ref), axis=1)
        else:
            q_large, k_large, v_large, b_large, a_large = (
                vmem_ldst.load_activation_as_large(
                    qkv_vreg=qkv_out_compact,
                    qkv_vmem_ref=qkv_work_ref,
                    b_vmem_ref=b_slot_ref,
                    a_vmem_ref=a_slot_ref,
                    cfgs=cfg,
                )
            )
        setup_scope.__exit__(None, None, None)

        with _scope("gdn_chunked"):
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
    store_scope = _scope("gdn_store")
    store_scope.__enter__()
    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    if new_recurrent_state is None:
        # Checkpoints were encoded inside the token loop.
        assert inline_ckpt or grouped
    elif cfg.state_plan is None:
        recurrent_out_slot_ref[...] = new_recurrent_state.astype(
            recurrent_out_slot_ref.dtype
        )
    else:
        first_ckpt = cfg.window_size - config.bench_ckpt_writes(cfg.window_size)
        for idx in range(cfg.seq_tile_size):
            for t in range(first_ckpt, cfg.window_size):
                args = (
                    recurrent_out_slot_ref.at[idx, t],
                    conv_state_out_slot_ref.at[idx, t],
                    cfg,
                    new_recurrent_state[idx, t],
                )
                if carry_recurrent_scratch_ref is None:
                    vmem_ldst.store_recurrent_state(*args)
                else:
                    pl.when(metadata_ref.p_id_is_last_tile[p_id, idx])(
                        functools.partial(vmem_ldst.store_recurrent_state, *args)
                    )

    if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]

    if manual_dma is not None:
        # Nothing runs after the last tile, so its own checkpoints and the
        # previous tile's are drained here.
        last = metadata_ref.num_tiles[...] - 1
        if deferred_dma:
            manual_dma.start_step(p_id, pl.when(p_id == last))
        manual_dma.wait_step(p_id - 1, pl.when((p_id == last) & (p_id >= 1)))
        manual_dma.wait_step(p_id, pl.when(p_id == last))
    store_scope.__exit__(None, None, None)


def _dense_conv_silu(
    *,
    qkv_slot_ref: jax.Ref,  # [1, aligned_chunk + 16, dim] bf16
    dense_ref: jax.Ref,  # [8 + aligned_chunk, dim] f32
    prev_conv: jax.Array,  # [1, prev_kernel_size, 1, dim] f32
    real_sizes: jax.Array,  # [1]
    conv_weight: jax.Array,  # [kernel_size, 1, dim] f32
    conv_bias: jax.Array | None,  # [dim] f32
    conv_state_out_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref,
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    cfg: config.GDNConfig,
) -> jax.Array:
    """Token-dense unpack, causal conv, SiLU and conv-state store of one
    PER_SEQ tile; returns the activated rows [chunk, dim] in f32.

    The compact layout keeps one token row per vreg (one of eight sublanes
    in use) so a tap is a whole-vreg op per 128 lanes per token; here the
    rows are dense and a tap is a sublane shift of an 8-row-aligned load.
    The conv state (rows [real_sizes, real_sizes + prev_kernel_size) of the
    carried-plus-chunk rows) is a static tail slice on a full tile and a
    masked row pick on the partial last tile of a sequence.
    """
    chunk = cfg.chunk_size
    kprev = cfg.prev_kernel_size
    dim = cfg.dim_size
    aligned_chunk = pl.cdiv(chunk, 16) * 16
    r_base = metadata_ref.p_id_to_r_base[p_id, 0]
    r_prefix = jax.lax.rem(r_base, 16)
    stub_conv = config.bench_stub() == "conv"

    with _scope("gdn_unpack"):
        for prefix in range(16):

            @pl.when(r_prefix == prefix)
            def _unpack(prefix=prefix):
                load_rows = pl.cdiv(prefix + aligned_chunk, 16) * 16
                rows = qkv_slot_ref[0, :load_rows, :]
                dense_ref[8 : 8 + aligned_chunk, :] = rows[
                    prefix : prefix + aligned_chunk, :
                ].astype(jnp.float32)

        for k in range(kprev):
            dense_ref[8 - kprev + k : 8 - kprev + k + 1, :] = prev_conv[0, k]

    with _scope("gdn_conv"):
        # [8 + chunk, dim]: row 8 - kprev + t is token t - kprev of the chunk.
        x = dense_ref[0 : 8 + chunk, :]
        if stub_conv:
            y = x[8 : 8 + chunk]
        else:
            y = x[8 - kprev : 8 - kprev + chunk] * conv_weight[0]
            for k in range(1, cfg.kernel_size):
                y = y + x[8 - kprev + k : 8 - kprev + k + chunk] * conv_weight[k]
            if conv_bias is not None:
                y = y + conv_bias.reshape(1, dim)
        y = jax.nn.silu(y)

        def _compact(rows2d: jax.Array) -> jax.Array:
            # [kprev, dim] dense rows as the state's [kprev, 1, dim] layout.
            return jnp.stack([rows2d[k : k + 1] for k in range(kprev)], axis=0)

        def _store_state(state3: jax.Array) -> None:
            _store_conv_state_tile(conv_state_out_slot_ref, cfg, state3)

        rs = real_sizes[0]
        is_last = metadata_ref.p_id_is_last_tile[p_id, 0]
        tail = _compact(x[8 + chunk - kprev : 8 + chunk])
        # The next tile of the sequence resumes from this tile's last rows.
        carry_conv_scratch_ref[0] = tail

        @pl.when(jnp.logical_and(is_last, rs == chunk))
        def _full():
            _store_state(tail)

        @pl.when(rs < chunk)
        def _partial():
            # Only a sequence's last tile is partial.
            row = jax.lax.broadcasted_iota(jnp.int32, (8 + chunk, 1), 0)
            picks = [
                jnp.sum(jnp.where(row == 8 - kprev + rs + k, x, 0.0), axis=0, keepdims=True)
                for k in range(kprev)
            ]
            _store_state(_compact(jnp.concatenate(picks, axis=0)))

    return y


def _store_conv_state_tile(conv_state_out_slot_ref: jax.Ref, cfg: config.GDNConfig, state3):
    """Stores one PER_SEQ tile's conv state ([prev_kernel_size, 1, dim]) into
    window position 0 of slot 0, through the plan's region view when pooled."""
    if cfg.state_plan is None:
        conv_state_out_slot_ref[0, 0] = state3
    else:
        vmem_ldst.store_state_region(
            conv_state_out_slot_ref.at[0, 0], cfg.state_plan.conv, state3
        )


def _lazy_conv_state(
    *,
    lhs: jax.Array,  # [prev_kernel_size + chunk, 1, dim] f32
    real_sizes: jax.Array,  # [1]
    conv_state_out_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref,
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    cfg: config.GDNConfig,
) -> None:
    """Conv state of a PER_SEQ chunked tile from its compact rows: the static
    tail on a full tile, the per-row masked select only on the partial last
    tile of a sequence."""
    chunk, kprev = cfg.chunk_size, cfg.prev_kernel_size
    rs = real_sizes[0]
    is_last = metadata_ref.p_id_is_last_tile[p_id, 0]
    tail = lhs[chunk : chunk + kprev]
    carry_conv_scratch_ref[0] = tail

    @pl.when(jnp.logical_and(is_last, rs == chunk))
    def _full():
        _store_conv_state_tile(conv_state_out_slot_ref, cfg, tail)

    @pl.when(rs < chunk)
    def _partial():
        _store_conv_state_tile(
            conv_state_out_slot_ref, cfg, compute_conv1d.conv_state_rows(lhs, rs, cfg)
        )


class _ManualCkptDma:
    """Write-back of recurrent checkpoints from the kernel body.

    Checkpoint t of tile slot idx is encoded into `stage[step % 2, idx, t]`
    and DMA'd to its source block right away; the staging slot is reused two
    steps later, after waiting on the copies it issued. One wait per copy,
    mirroring the start descriptors, as the pipeline's buffered refs do.
    """

    def __init__(self, *, metadata_ref, p_id, stage_ref, sems, source_ref, cfg):
        self.metadata_ref = metadata_ref
        self.p_id = p_id
        self.stage_ref = stage_ref
        self.sems = sems
        self.source_ref = source_ref
        self.cfg = cfg
        self.region = cfg.state_plan.recurrent
        self.stride = cfg.state_plan.stride
        self.first_ckpt = cfg.window_size - config.bench_ckpt_writes(cfg.window_size)
        self.slot = jax.lax.rem(p_id, 2)
        self.stage = stage_ref.at[self.slot]

    def _nblocks(self, step, idx: int, t: int):
        md = self.metadata_ref
        is_last_tile = md.p_id_is_last_tile[step, idx]
        num_ckpts = jnp.minimum(md.p_id_to_r_size[step, idx], self.cfg.window_size)
        should_write = jnp.logical_and(is_last_tile, t < num_ckpts)
        return jnp.where(should_write, self.region.nblocks, 0)

    def start(self, idx: int, t: int, step=None) -> None:
        """Starts the copy of checkpoint t of slot idx, of this step by
        default, from the staging slot that step encoded into."""
        md = self.metadata_ref
        if step is None:
            step, slot = self.p_id, self.slot
        else:
            slot = jax.lax.rem(step, 2)
        nblocks = self._nblocks(step, idx, t)
        s_idx = md.p_id_to_s_idx[step, idx]
        block = md.s_idx_to_ckpt_indices[s_idx, t] * self.stride + self.region.kb0
        pltpu.make_async_copy(
            self.stage_ref.at[slot, idx, t, pl.ds(0, nblocks)],
            self.source_ref.at[pl.ds(block, nblocks)],
            self.sems.at[slot],
        ).start()

    def start_step(self, step, when) -> None:
        """Starts every checkpoint copy tile `step` encoded into its slot."""

        @when
        def _():
            for idx in range(self.cfg.seq_tile_size):
                for t in range(self.first_ckpt, self.cfg.window_size):
                    self.start(idx, t, step)

    def wait_step(self, step, when) -> None:
        """Waits for every copy tile `step` issued from its staging slot."""
        slot = jax.lax.rem(step, 2)

        @when
        def _():
            for idx in range(self.cfg.seq_tile_size):
                for t in range(self.first_ckpt, self.cfg.window_size):
                    nblocks = self._nblocks(step, idx, t)
                    ref = self.stage_ref.at[slot, idx, t, pl.ds(0, nblocks)]
                    pltpu.make_async_copy(ref, ref, self.sems.at[slot]).wait()


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
    ckpt_stage_ref: jax.Array | None = None,
    ckpt_sems: jax.Array | None = None,
    dense_act_scratch_ref: jax.Array | None = None,
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
                act_scratch_ref,
                ckpt_stage_ref,
                ckpt_sems,
                dense_act_scratch_ref,
            ),
            allocations=allocations,
        )

    @pl.when(num_tiles > 0)
    def _():
        _run()


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state", "state_source"),
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
        "state_plan",
        "attention_mode",
        "gate_lower_bound",
        "bf16_qkv_mode",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
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
    batched_only: bool = False,
    prefill_only: bool = False,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    # TODO(kyuyeunk): Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int = 128,
    seg_bounds: jax.Array | None = None,  # [num_seqs, 2]
    seg_state_indices: jax.Array | None = None,  # [num_seqs, 2]
    bf16_qkv_mode: int | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array] | tuple[jax.Array, jax.Array]:
    """Execute depthwise causal Conv1d, SiLU activation, L2 normalization, and
    recurrent Gated DeltaNet state update in a single fused Pallas kernel.

    Supports variable-length prefill chunks, speculative decode verification
    windows, and single-token decode steps over pooled conv and SSM state caches.

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
        decode_tile_size: Tile size along sequence dimension for decode sequences,
            capped so its 1 + window state tiles fit the decode VMEM budget.
            Replaced by the dynamic tiler's pick when
            TPU_ENABLE_GDN_DYNAMIC_TILING is set (GDN only), and by
            TPU_GDN_DECODE_TILE_SIZE when that is positive.
        mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
            sequences: capped at 64 under KDA and rounded down to a multiple
            of 16 above 16. Under GDN it is further capped to the largest
            chunk whose VMEM estimate fits.
        seg_bounds: Optional [num_seqs, 2] query offsets, prefill/mixed
            sequences only. Up to two extra state checkpoints are stored inside
            a sequence's chunk: the state after query offset `seg_bounds[s, k]`
            is written to `seg_state_indices[s, k]`. See
            `metadata.compute_per_seq_segment_metadata`.
        seg_state_indices: Optional [num_seqs, 2] state slots (dense) or source
            blocks (pooled) receiving those checkpoints. Required with
            `seg_bounds`.
        bf16_qkv_mode: How a bf16 qkv reaches the kernel: 0 casts it to f32
            rows, 1 streams [T, 2, dim / 2] bf16, 2 streams the 2D [T, dim]
            bf16 rows in 16-row aligned DMAs. None reads TPU_GDN_BF16_QKV
            (default 2). Modes 1 and 2 need dim a multiple of 2 * num_lanes,
            at least 16 tokens and a multiple-of-8 mixed chunk, and fall back
            to 0 otherwise.

    Returns:
        (new_conv_state, new_recurrent_state): Updated convolution state cache and
            recurrent state cache tensors. With a state source: the updated
            source (aliased in place) instead of the state pair.
        out: Fused output tensor.
    """
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

    assert a.dtype == b.dtype == qkv.dtype

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
        read_offsets = jnp.zeros((num_seqs,), dtype=jnp.int32)
    assert read_offsets.shape == (num_seqs,)
    read_offsets = read_offsets.astype(jnp.int32)
    if num_spec_tokens > 0 and ckpt_indices is not None:
        read_offsets = jnp.clip(read_offsets, 0, num_spec_tokens)
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
    if seg_bounds is not None:
        assert seg_state_indices is not None
        assert seg_bounds.shape == (num_seqs, 2), seg_bounds.shape
        assert seg_state_indices.shape == (num_seqs, 2), seg_state_indices.shape
        # A pooled source without speculative decoding addresses checkpoint 0
        # through state_indices alone; segments need separate read and write
        # blocks, so give PER_SEQ a one-column checkpoint table.
        seg_ckpt_indices = ckpt_indices
        if pooled and seg_ckpt_indices is None:
            seg_ckpt_indices = state_indices.astype(jnp.int32)[:, None]

    num_lanes = pltpu.get_tpu_info().num_lanes
    decode_tile_size = min(decode_tile_size, batch_size)
    if is_kda:
        mixed_tile_size = min(mixed_tile_size, 64)
    mixed_tile_size = min(mixed_tile_size, batch_size)
    if mixed_tile_size > 16:
        mixed_tile_size = (mixed_tile_size // 16) * 16
    if bf16_qkv_mode is None:
        bf16_qkv_mode = _bf16_qkv_mode()
    if (
        bf16_qkv_mode in (1, 2)
        and qkv.dtype == jnp.bfloat16
        and dim % (2 * num_lanes) == 0
        and batch_size >= 16
        and mixed_tile_size % 8 == 0
    ):
        qkv_2d = bf16_qkv_mode == 2
        act_in_dtype = qkv.dtype
    else:
        bf16_qkv_mode = 0
        qkv_2d = False
        act_in_dtype = jnp.dtype(jnp.float32)

    packing = 16 if qkv_2d else (4 // act_in_dtype.itemsize)
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes
    window_size = num_spec_tokens + 1
    decode_vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(window_size)

    rec_region_bytes: int | None = None
    conv_region_bytes: int | None = None
    if state_plan is not None:
        assert state_source is not None
        rec_region_bytes = _pooled_region_bytes(
            state_source, state_plan, state_plan.recurrent
        )
        conv_region_bytes = _pooled_region_bytes(
            state_source, state_plan, state_plan.conv
        )

    # KDA stays on static tiles until the VMEM estimate models its gate.
    dynamic_tiling = envs.TPU_ENABLE_GDN_DYNAMIC_TILING and not is_kda
    dyn_decode_tile: int | None = None
    if not is_kda:
        # GDN caps the prefill/mixed chunk at its VMEM fit in both modes. The
        # flag selects the decode tile: the tiler's pick, or decode_tile_size
        # under spec_tile_cap below.
        fit_decode_tile, mixed_tile_size = tiling.get_tile_sizes(
            batch_size=batch_size,
            num_seqs=num_seqs,
            padded_batch_size=padded_batch_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim,
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=(
                jnp.float32 if recurrent_state is None else recurrent_state.dtype
            ),
            num_lanes=num_lanes,
            decode_vmem_limit_bytes=decode_vmem_limit_bytes,
            mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
            window_size=window_size,
            mixed_tile_size=mixed_tile_size,
            rec_region_bytes=rec_region_bytes,
            conv_region_bytes=conv_region_bytes,
            qkv_2d=qkv_2d,
        )
        if dynamic_tiling:
            dyn_decode_tile = fit_decode_tile
        chunk_override = config.prefill_chunk()
        if chunk_override > 0:
            mixed_tile_size = max(1, min(chunk_override, batch_size))
        if bf16_qkv_mode and mixed_tile_size % 8:
            # The bf16 qkv layouts run only on multiple-of-8 chunks, and the
            # VMEM fit can cap the chunk below the one checked above.
            # padded_batch_size stays a multiple of 16, which f32 rows accept.
            bf16_qkv_mode = 0
            qkv_2d = False
            act_in_dtype = jnp.dtype(jnp.float32)

    spec_tile_cap = tiling.spec_window_tile_cap(
        window=window_size,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        dim=dim,
        aligned_num_v_heads=aligned_num_v_heads,
        num_buffers=config.GDNConfig.__dataclass_fields__["num_buffers"].default,
        vmem_budget=decode_vmem_limit_bytes,
        rec_region_bytes=rec_region_bytes,
        conv_region_bytes=conv_region_bytes,
    )
    decode_tile_size = tiling.select_decode_tile_size(
        default_tile=decode_tile_size,
        dyn_tile=dyn_decode_tile,
        batch_size=batch_size,
        num_spec_tokens=num_spec_tokens,
        spec_tile_cap=spec_tile_cap,
        is_kda=is_kda,
        override=_decode_tile_override(),
        num_seqs=num_seqs,
    )
    if dynamic_tiling:
        logger.info_once(
            "GDN dynamic tiling: num_seqs=%d batch=%d window=%d -> T=%d C=%d",
            num_seqs,
            batch_size,
            window_size,
            decode_tile_size,
            mixed_tile_size,
        )

    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    # KDA's gate carries d_k channels per value head, so it pads out to a
    # lane multiple of n_v * d_k rather than of n_v.
    gate_dim = n_v * d_k if is_kda else n_v
    gate_padding_size = (pl.cdiv(gate_dim, num_lanes) * num_lanes) - gate_dim
    if batch_padding_size:
        qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
    if bf16_qkv_mode == 1:
        qkv = qkv.reshape(padded_batch_size, 2, dim // 2)
    elif bf16_qkv_mode == 0:
        qkv = qkv.reshape(padded_batch_size, 1, -1).astype(jnp.float32)
    if not is_kda:
        # Fuse b and a [N, n_v] -> [2, N_pad, 128] into a single HBM pad where
        # both leading-index slices remain 100% contiguous [N_pad, 1, 128].
        ba_padded = jnp.pad(
            jnp.stack((b, a), axis=0),
            ((0, 0), (0, batch_padding_size), (0, num_v_padding_size)),
        )
        b = ba_padded[0].reshape(padded_batch_size, 1, -1).astype(jnp.float32)
        a = ba_padded[1].reshape(padded_batch_size, 1, -1).astype(jnp.float32)
        a_log = jnp.pad(a_log, ((0, num_v_padding_size)))
        dt_bias = jnp.pad(dt_bias, ((0, num_v_padding_size)))
    else:
        if batch_padding_size or num_v_padding_size:
            b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
        if batch_padding_size or gate_padding_size:
            a = jnp.pad(a, ((0, batch_padding_size), (0, gate_padding_size)))
        b = b.reshape(padded_batch_size, 1, -1).astype(jnp.float32)
        a = a.reshape(padded_batch_size, 1, -1).astype(jnp.float32)

    # Step 3: States and weights pre-processing.
    # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
    # loading stage.
    if not pooled:
        conv_state_shape = cast(jax.Array, conv_state).shape
        conv_state = cast(jax.Array, conv_state).reshape(-1, kernel_size - 1, 1, dim)
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
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1

        cfg = config.GDNConfig(
            mode=mode,
            batch_size=padded_batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            window_size=window_size,
            qkv_2d=qkv_2d,
            dim_size=dim,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d_k,
            v_head_dim=d_v,
            attention_mode=attention_mode,
            gate_lower_bound=gate_lower_bound,
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
            manual_ckpt_dma=(
                config.ckpt_manual_dma()
                and pooled
                and mode == config.GDNMode.BATCHED
                and not is_kda
                and window_size > 0
            ),
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
            if seg_bounds is not None:
                metadata_obj = metadata.compute_per_seq_segment_metadata(
                    cfg=cfg,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    state_indices=state_indices,
                    read_offsets=read_offsets,
                    start_seq=distribution[0],
                    end_seq=distribution[-1],
                    seg_bounds=seg_bounds,
                    seg_state_indices=seg_state_indices,
                    ckpt_indices=seg_ckpt_indices,
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
        vmem_fraction = config.prefill_vmem_fraction()
        if mode == config.GDNMode.PER_SEQ and vmem_fraction > 0:
            vmem_limit_bytes = int(
                vmem_fraction * pltpu.get_tpu_info().vmem_capacity_bytes
            )

        scratch_shapes = cfg.get_scratch_shape_dict()
        scratch_shapes.update(ckpt_stage_ref=None, ckpt_sems=None)
        if cfg.manual_ckpt_dma:
            assert state_plan is not None
            scratch_shapes.update(
                ckpt_stage_ref=pltpu.VMEM(
                    (
                        2,
                        cfg.seq_tile_size,
                        cfg.window_size,
                        state_plan.recurrent.nblocks,
                        *in_conv_state.shape[1:],
                    ),
                    in_conv_state.dtype,
                ),
                ckpt_sems=pltpu.SemaphoreType.DMA((2,)),
            )

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
            # A prefill-only runner always has a PER_SEQ range and calls it
            # directly (kernels/static_guards.py).
            if (
                not prefill_only
                and out_act is not None
                and not static_guards.prefill_only()
                and os.environ.get("TPU_GDN_SKIP_PER_SEQ", "cond") == "cond"
            ):
                out_act, out_source = jax.lax.cond(
                    distribution[0] < distribution[2],
                    lambda os_pair: call_kernel(
                        os_pair[1], None, os_pair[0], config.GDNMode.PER_SEQ
                    )[:2],
                    lambda os_pair: os_pair,
                    (out_act, out_source),
                )
            else:
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
