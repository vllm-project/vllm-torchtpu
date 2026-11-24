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
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


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
        cfg: GDN configuration object.
        p_id: Optional explicit tile index. Communication-fused outer kernels
            pass their global compute-tile index because their surrounding
            pipeline owns the Pallas program id.
    """

    if p_id is None:
        p_id = pl.program_id(0)

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
    # Round to the state dtype, then pack into the uint32 window.
    recurrent_out_slot_ref[...] = pltpu.bitcast(
        new_recurrent_state.astype(cfg.dtypes.recurrent_state), jnp.uint32
    )

    if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]


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


def _conv_silu_heads_bf16(
    S4: jax.Array,  # bf16[4C, K_ext_pad]
    X_ext: jax.Array,  # bf16[K_ext_pad, dim]
    conv_weight_ref: jax.Ref,  # f32[4, 1, dim]
    conv_bias_all: jax.Array | None,  # f32[1, dim]
    col_start: int,
    n_cols: int,
    C: int,
    N_blk: int,
    head_dim: int,
    full_carry_parts: list[jax.Array],
) -> jax.Array:
    """N-tiled MXU Toeplitz Shift Conv1D + SiLU returning f32[n_heads, C, head_dim]."""
    assert n_cols % N_blk == 0 and N_blk % head_dim == 0
    heads_per_blk = N_blk // head_dim
    head_list: list[jax.Array] = []
    for col in range(col_start, col_start + n_cols, N_blk):
        X_blk = X_ext[:, col : col + N_blk]  # bf16[K_ext_pad, N_blk]
        sh = jax.lax.dot_general(
            S4,
            X_blk,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )  # f32[4C, N_blk]
        c0 = sh[0:C]
        c1 = sh[C : 2 * C]
        c2 = sh[2 * C : 3 * C]
        c3 = sh[3 * C : 4 * C]
        full_carry_parts.append(c0[C - 16 : C].astype(jnp.bfloat16))

        w_blk = conv_weight_ref[:, :, col : col + N_blk].astype(jnp.float32)
        conv = c0 * w_blk[3] + c1 * w_blk[2] + c2 * w_blk[1] + c3 * w_blk[0]
        if conv_bias_all is not None:
            conv = conv + conv_bias_all[:, col : col + N_blk]
        act = jax.nn.silu(conv)  # f32[C, N_blk]
        for h in range(heads_per_blk):
            head_list.append(act[:, h * head_dim : (h + 1) * head_dim])
    return jnp.stack(head_list, axis=0)


def _inner_kernel_per_seq_bf16(
    qkv_slot_ref: jax.Ref,  # bf16[max_blks, 16, dim]
    b_slot_ref: jax.Ref,
    a_slot_ref: jax.Ref,
    conv_state_in_slot_ref: jax.Ref,  # [1, 1, 3, 1, dim]
    recurrent_in_slot_ref: jax.Ref,  # [1, 1, n_v, d_k // packing, d_v] uint32
    out_slot_ref: jax.Ref,  # [1, C, n_v, d_v]
    conv_state_out_slot_ref: jax.Ref,  # [1, 1, 3, 1, dim]
    recurrent_out_slot_ref: jax.Ref,  # [1, 1, n_v, d_k // packing, d_v] uint32
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Ref,  # bf16[16, dim]
    carry_recurrent_scratch_ref: jax.Ref,  # f32[1, n_v, d_k, d_v]
    *,
    cfg: config.GDNConfig,
    head_groups: int,
    conv_n_block: int,
):
    """PER_SEQ inner kernel reading native bf16[T//16, 16, dim] via MXU Toeplitz Conv1D."""
    p_id = pl.program_id(0)
    s_idx = metadata_ref.p_id_to_s_idx[p_id, 0]
    r_base = metadata_ref.p_id_to_r_base[p_id, 0]
    r_size = metadata_ref.p_id_to_r_size[p_id, 0]
    is_first_tile = metadata_ref.p_id_is_first_tile[p_id, 0]
    is_last_tile = metadata_ref.p_id_is_last_tile[p_id, 0]
    has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]
    if metadata_ref.s_idx_to_prefill_ckpt_row is not None:
        should_store_state = is_last_tile | metadata_ref.is_prefill_ckpt_tile(p_id, 0)
    else:
        should_store_state = is_last_tile

    C = cfg.chunk_size
    dim = cfg.dim_size
    n_kq = cfg.num_kq_heads
    n_v = cfg.num_v_heads
    d_k = cfg.kq_head_dim
    d_v = cfg.v_head_dim
    max_blks = min(cfg.batch_size // 16, C // 16 + 1)
    K_ext = 16 + max_blks * 16
    K_ext_pad = pl.cdiv(K_ext, 128) * 128

    @pl.when(is_first_tile)
    def _load_init():
        raw_c = conv_state_in_slot_ref[0, 0].astype(jnp.float32)
        raw_r = pltpu.bitcast(
            recurrent_in_slot_ref[0, 0], cfg.dtypes.recurrent_state
        ).astype(jnp.float32)
        c3 = jnp.where(has_initial_state, raw_c, 0.0)
        row16 = jax.lax.broadcasted_iota(jnp.int32, (16, dim), 0)
        prev16 = jnp.where(
            row16 == 13,
            c3[0],
            jnp.where(row16 == 14, c3[1], jnp.where(row16 == 15, c3[2], 0.0)),
        ).astype(jnp.bfloat16)
        carry_conv_scratch_ref[...] = prev16
        carry_recurrent_scratch_ref[0] = jnp.where(has_initial_state, raw_r, 0.0)

    prev_16 = carry_conv_scratch_ref[...]
    state_prev = carry_recurrent_scratch_ref[0]

    # NaN-proof zero-masking of unread VMEM blocks and in-block padding rows.
    r_off = r_base & 15
    raw_qkv = qkv_slot_ref[...].reshape(max_blks * 16, dim)
    qrow = jax.lax.broadcasted_iota(jnp.int32, (max_blks * 16, dim), 0)
    valid_row = (qrow >= r_off) & (qrow < r_off + r_size)
    qkv_clean = jnp.where(valid_row, raw_qkv, jnp.bfloat16(0))
    if K_ext_pad > K_ext:
        X_ext = jnp.concat(
            [prev_16, qkv_clean, jnp.zeros((K_ext_pad - K_ext, dim), jnp.bfloat16)],
            axis=0,
        )
    else:
        X_ext = jnp.concat([prev_16, qkv_clean], axis=0)

    # Exact 0/1 Toeplitz shift matrix S4: bf16[4C, K_ext_pad] (128-lane aligned).
    t_idx = jax.lax.broadcasted_iota(jnp.int32, (C, K_ext_pad), 0)
    k_idx = jax.lax.broadcasted_iota(jnp.int32, (C, K_ext_pad), 1)
    S_parts = []
    for d in (0, 1, 2, 3):
        src_col = jnp.where(t_idx - d >= 0, 16 + r_off + t_idx - d, 16 + t_idx - d)
        S_parts.append((k_idx == src_col).astype(jnp.bfloat16))
    S4 = jnp.concat(S_parts, axis=0)

    conv_w_ref = weights_ref.conv.weight
    conv_b_all = (
        weights_ref.conv.bias[...].astype(jnp.float32).reshape(1, dim)
        if weights_ref.conv.bias is not None
        else None
    )
    full_carry_parts: list[jax.Array] = []

    mask_dtype = compute_gdn.get_mask_dtype(cfg.dtypes.compute)
    iota_c = jax.lax.broadcasted_iota(mask_dtype, (1, C, 1), 1)
    mask_h = iota_c < r_size.astype(mask_dtype)

    qk_cols = n_kq * d_k
    q_kq = _conv_silu_heads_bf16(
        S4,
        X_ext,
        conv_w_ref,
        conv_b_all,
        0,
        qk_cols,
        C,
        conv_n_block,
        d_k,
        full_carry_parts,
    )
    k_kq = _conv_silu_heads_bf16(
        S4,
        X_ext,
        conv_w_ref,
        conv_b_all,
        qk_cols,
        qk_cols,
        C,
        conv_n_block,
        d_k,
        full_carry_parts,
    )
    q_kq = jnp.where(mask_h, q_kq.astype(cfg.dtypes.compute), 0.0)
    k_kq = jnp.where(mask_h, k_kq.astype(cfg.dtypes.compute), 0.0)
    q_kq = compute_gdn.l2_norm(q_kq) * (d_k**-0.5)
    k_kq = compute_gdn.l2_norm(k_kq)

    b_large = jnp.expand_dims(vmem_ldst.load_compact_to_large(b_slot_ref), axis=1)[0]
    a_large = jnp.expand_dims(vmem_ldst.load_compact_to_large(a_slot_ref), axis=1)[0]
    padding_size = cfg.aligned_num_v_heads - n_v
    a_log = (
        jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
        .reshape(1, 1, -1)
        .astype(cfg.dtypes.compute)
    )
    dt_bias = (
        jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))
        .reshape(1, 1, -1)
        .astype(cfg.dtypes.compute)
    )

    beta_s = jax.nn.sigmoid(b_large.astype(cfg.dtypes.compute))
    g_log_s = -jnp.exp(a_log) * jax.nn.softplus(
        a_large.astype(cfg.dtypes.compute) + dt_bias
    )
    beta_s = jnp.where(mask_h, beta_s, 0.0)
    g_log_s = jnp.where(mask_h, g_log_s, 0.0)

    g_log2_2d = (g_log_s[0] * jnp.float32(compute_gdn.LOG2E)).astype(jnp.float32)
    r_c = jax.lax.broadcasted_iota(jnp.int32, (C, C), 0)
    c_c = jax.lax.broadcasted_iota(jnp.int32, (C, C), 1)
    tril_f32 = (r_c >= c_c).astype(jnp.float32)
    g_cs_2d = jax.lax.dot_general(
        tril_f32,
        g_log2_2d,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    g_cs_all = compute_gdn.fused_transpose_broadcast(
        g_cs_2d[None, :, :], src_dim=2, dst_dim=0, dst_size=n_v
    )
    beta_all = compute_gdn.fused_transpose_broadcast(
        beta_s, src_dim=2, dst_dim=0, dst_size=n_v
    )

    def _finish_conv_carry():
        full_carry_16 = jnp.concat(full_carry_parts, axis=-1)

        def _compute_tail_carry(_):
            u_idx = jax.lax.broadcasted_iota(jnp.int32, (16, K_ext_pad), 0)
            k_tail = jax.lax.broadcasted_iota(jnp.int32, (16, K_ext_pad), 1)
            d_tail = u_idx - 16
            tail_col = jnp.where(
                r_size > 0,
                jnp.where(
                    r_size + d_tail >= 0,
                    16 + r_off + r_size + d_tail,
                    16 + r_size + d_tail,
                ),
                u_idx,
            )
            S_tail = ((u_idx >= 13) & (k_tail == tail_col)).astype(jnp.bfloat16)
            tail_parts = [
                jax.lax.dot_general(
                    S_tail,
                    X_ext[:, col : col + conv_n_block],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                for col in range(0, dim, conv_n_block)
            ]
            return jnp.concat(tail_parts, axis=-1)

        new_prev_16 = jax.lax.cond(
            r_size < C, _compute_tail_carry, lambda _: full_carry_16, None
        )
        carry_conv_scratch_ref[...] = new_prev_16

        @pl.when(should_store_state)
        def _store_final_conv():
            row16 = jax.lax.broadcasted_iota(jnp.int32, (16, dim), 0)
            np_f32 = new_prev_16.astype(jnp.float32)
            c0 = jnp.sum(jnp.where(row16 == 13, np_f32, 0.0), axis=0, keepdims=True)
            c1 = jnp.sum(jnp.where(row16 == 14, np_f32, 0.0), axis=0, keepdims=True)
            c2 = jnp.sum(jnp.where(row16 == 15, np_f32, 0.0), axis=0, keepdims=True)
            new_conv_3 = jnp.stack([c0, c1, c2], axis=0)
            conv_state_out_slot_ref[0, 0] = new_conv_3.astype(
                conv_state_out_slot_ref.dtype
            )

    v_col_base = 2 * qk_cols
    if head_groups == 1:
        v_all = _conv_silu_heads_bf16(
            S4,
            X_ext,
            conv_w_ref,
            conv_b_all,
            v_col_base,
            n_v * d_v,
            C,
            conv_n_block,
            d_v,
            full_carry_parts,
        )
        _finish_conv_carry()
        v_all = jnp.where(mask_h, v_all.astype(cfg.dtypes.compute), 0.0)
        out_all, new_state = compute_gdn.chunked_gdn_group(
            q_kq, k_kq, v_all, g_cs_all, beta_all, state_prev, cfg=cfg
        )
        out_all_bf16 = out_all.astype(out_slot_ref.dtype)
    else:
        assert n_kq % head_groups == 0 and n_v % head_groups == 0
        kq_per = n_kq // head_groups
        v_per = n_v // head_groups
        out_parts: list[jax.Array] = []
        state_parts: list[jax.Array] = []
        for g_idx in range(head_groups):
            kq0, kq1 = g_idx * kq_per, (g_idx + 1) * kq_per
            v0, v1 = g_idx * v_per, (g_idx + 1) * v_per
            v_grp = _conv_silu_heads_bf16(
                S4,
                X_ext,
                conv_w_ref,
                conv_b_all,
                v_col_base + v0 * d_v,
                v_per * d_v,
                C,
                conv_n_block,
                d_v,
                full_carry_parts,
            )
            v_grp = jnp.where(mask_h, v_grp.astype(cfg.dtypes.compute), 0.0)
            out_g, st_g = compute_gdn.chunked_gdn_group(
                q_kq[kq0:kq1],
                k_kq[kq0:kq1],
                v_grp,
                g_cs_all[v0:v1],
                beta_all[v0:v1],
                state_prev[v0:v1],
                cfg=cfg,
            )
            out_parts.append(out_g.astype(out_slot_ref.dtype))
            state_parts.append(st_g)
        out_all_bf16 = jnp.concat(out_parts, axis=0)
        new_state = jnp.concat(state_parts, axis=0)
        _finish_conv_carry()

    out_slot_ref[0] = out_all_bf16.swapaxes(0, 1)
    carry_recurrent_scratch_ref[0] = new_state

    @pl.when(should_store_state)
    def _store_rec():
        recurrent_out_slot_ref[0, 0] = pltpu.bitcast(
            new_state.astype(cfg.dtypes.recurrent_state), jnp.uint32
        )


def _inner_kernel_batched_bf16(
    qkv_slot_ref: jax.Ref,  # bf16[seq_tile_size, 16, dim]
    b_slot_ref: jax.Ref,
    a_slot_ref: jax.Ref,
    conv_state_in_slot_ref: jax.Ref,
    recurrent_in_slot_ref: jax.Ref,
    out_slot_ref: jax.Ref,
    conv_state_out_slot_ref: jax.Ref,
    recurrent_out_slot_ref: jax.Ref,
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    *,
    cfg: config.GDNConfig,
):
    """BATCHED inner kernel reading native bf16[T//16, 16, dim] with 1-block dedup."""
    p_id = pl.program_id(0)
    real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_in_slot_ref=conv_state_in_slot_ref,
        recurrent_in_slot_ref=recurrent_in_slot_ref,
        carry_conv_scratch_ref=carry_conv_scratch_ref,
        carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
        cfg=cfg,
    )

    r_base_0 = metadata_ref.p_id_to_r_base[p_id, 0]
    r_size_0 = metadata_ref.p_id_to_r_size[p_id, 0]
    blk_0 = r_base_0 // 16
    need_0 = r_size_0 > 0
    row8 = jax.lax.broadcasted_iota(jnp.int32, (8, cfg.dim_size), 0)
    row8_hi = row8 + 8

    def _extract_from_blk(
        top8: jax.Array, bot8: jax.Array, r_off: jax.Array, r_size: jax.Array
    ) -> jax.Array:
        valid_s = r_size > 0
        m_top = (row8 == r_off) & valid_s
        m_bot = (row8_hi == r_off) & valid_s
        clean8 = jnp.where(m_top, top8, jnp.where(m_bot, bot8, 0.0))
        return jnp.expand_dims(jnp.sum(clean8, axis=0, keepdims=True), axis=0)

    def _fast_single_blk(_):
        blk0_f32 = qkv_slot_ref[0].astype(jnp.float32)
        top8_0 = blk0_f32[:8]
        bot8_0 = blk0_f32[8:]
        tok_list = []
        for idx in range(cfg.seq_tile_size):
            r_off = metadata_ref.p_id_to_r_base[p_id, idx] & 15
            r_size = metadata_ref.p_id_to_r_size[p_id, idx]
            tok_list.append(_extract_from_blk(top8_0, bot8_0, r_off, r_size))
        return jnp.stack(tok_list, axis=0)

    if cfg.seq_tile_size == 1:
        qkv_compact = _fast_single_blk(None)
    else:
        all_same = need_0
        for idx in range(1, cfg.seq_tile_size):
            rb_i = metadata_ref.p_id_to_r_base[p_id, idx]
            rs_i = metadata_ref.p_id_to_r_size[p_id, idx]
            all_same = all_same & ((rs_i == 0) | ((rb_i // 16) == blk_0))

        def _general_multi_blk(_):
            blk0_f32 = qkv_slot_ref[0].astype(jnp.float32)
            top8_0 = blk0_f32[:8]
            bot8_0 = blk0_f32[8:]
            tok_list = [
                _extract_from_blk(top8_0, bot8_0, r_base_0 & 15, r_size_0)
            ]
            always_true = row8 >= 0
            for idx in range(1, cfg.seq_tile_size):
                rb_i = metadata_ref.p_id_to_r_base[p_id, idx]
                rs_i = metadata_ref.p_id_to_r_size[p_id, idx]
                same_i = ((rb_i // 16) == blk_0) & need_0
                same_mask = always_true & same_i
                blk_i_f32 = qkv_slot_ref[idx].astype(jnp.float32)
                top8_i = jnp.where(same_mask, top8_0, blk_i_f32[:8])
                bot8_i = jnp.where(same_mask, bot8_0, blk_i_f32[8:])
                tok_list.append(_extract_from_blk(top8_i, bot8_i, rb_i & 15, rs_i))
            return jnp.stack(tok_list, axis=0)

        qkv_compact = jax.lax.cond(
            all_same, _fast_single_blk, _general_multi_blk, None
        )

    qkv_lhs = jnp.concat([prev_conv, qkv_compact], axis=1)
    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    conv_bias = (
        weights_ref.conv.bias[...].astype(jnp.float32)
        if weights_ref.conv.bias is not None
        else None
    )
    qkv_out, new_conv_state = compute_conv1d.causal_conv1d(
        real_sizes=real_sizes,
        lhs=qkv_lhs,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        cfg=cfg,
    )
    conv_state_out_slot_ref[...] = new_conv_state.astype(conv_state_out_slot_ref.dtype)

    qkv_act = jax.nn.silu(qkv_out)
    padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
    a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))
    q_compact, k_compact, v_compact, b_compact, a_compact = (
        vmem_ldst.load_activation_as_compact(
            qkv_vreg=qkv_act,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )
    out, new_recurrent_state = compute_gdn.recurrent_gdn(
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

    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    recurrent_out_slot_ref[...] = pltpu.bitcast(
        new_recurrent_state.astype(cfg.dtypes.recurrent_state), jnp.uint32
    )


def outer_kernel_bf16(
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    states_ref: jax.Array | tuple[jax.Array, jax.Array],
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    out_ref: jax.Array,
    states_out_ref: jax.Array | tuple[jax.Array, jax.Array],
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
    head_groups: int,
    conv_n_block: int,
):
    """Outer pipeline for native bf16[T//16, 16, dim] QKV input."""
    if cfg.is_shared:
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
    ) = memory_ref.create_allocs_bf16_qkv(
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
    if cfg.mode == config.GDNMode.PER_SEQ:
        body = functools.partial(
            _inner_kernel_per_seq_bf16,
            cfg=cfg,
            head_groups=head_groups,
            conv_n_block=conv_n_block,
        )
    else:
        body = functools.partial(_inner_kernel_batched_bf16, cfg=cfg)

    scratches = (
        metadata_ref,
        weights_ref,
        carry_conv_scratch_ref,
        carry_recurrent_scratch_ref,
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


FUSED_CONV1D_GDN_STATIC_ARGNAMES: tuple[str, ...] = (
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
    "recurrent_dtype",
    "conv_dtype",
    "attention_mode",
    "gate_lower_bound",
    "beta_scale",
    "kernel_act_dtype",
    "kernel_conv_weight_dtype",
)


@jax.jit(
    donate_argnames=("states",),
    static_argnames=FUSED_CONV1D_GDN_STATIC_ARGNAMES,
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
    beta_scale: float = 1.0,
    recurrent_dtype: jnp.dtype | None = None,
    conv_dtype: jnp.dtype | None = None,
    num_spec_tokens: int = 0,
    batched_only: bool = False,
    prefill_only: bool = False,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    kernel_act_dtype: jnp.dtype = jnp.float32.dtype,
    kernel_conv_weight_dtype: jnp.dtype = jnp.float32.dtype,
    # TODO(kyuyeunk): Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int | None = None,
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
        beta_scale: KDA only. Scales the delta-rule step size,
            `beta = beta_scale * sigmoid(b)`. 1.0 (Kimi) keeps beta in
            [0, 1]; OLMo 3.5 uses 2.0 (allow_neg_eigval).
        num_spec_tokens: Number of speculative draft tokens (0 gives the
            plain 1-token-per-sequence decode path).
        batched_only: Run only the leading decode/verify segment.
        prefill_only: Run only PER_SEQ, leaving decode/verify checkpoints
            untouched. Mutually exclusive with batched_only.
        zero_initialize_out: Whether to zero-initialize the output buffer before
            executing non-batched sequences.
        compute_precision: Computation precision dtype.
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
        assert recurrent_dtype is not None
    else:
        assert len(states) == 2
        recurrent_dtype = states[1].dtype
        conv_dtype = states[0].dtype
    # Conv1D reads the compact layout, which needs fp32. The recurrent state
    # may be bf16: it is upcast once on load and rounded once on store.
    if jnp.dtype(conv_dtype) != jnp.float32 or jnp.dtype(recurrent_dtype) not in (
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.bfloat16),
    ):
        raise NotImplementedError(
            "GDN v3 needs an fp32 conv state and an fp32 or bf16 recurrent "
            f"state; got conv={conv_dtype} recurrent={recurrent_dtype}."
        )

    # Step 1: Validate inputs.
    is_kda = attention_mode == config.AttentionMode.KDA
    qkv_input_dtype = qkv.dtype
    b = b.astype(kernel_act_dtype)
    a = a.astype(kernel_act_dtype)

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

    if beta_scale != 1.0 and not is_kda:
        # GDN's delta rule has the same beta; only the KDA path scales it.
        raise NotImplementedError(
            "beta_scale is only implemented for KDA; got "
            f"attention_mode={attention_mode}."
        )
    if beta_scale <= 0:
        raise ValueError(f"beta_scale must be positive; got {beta_scale}.")

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

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    num_sublanes = tpu_info.num_sublanes
    # The unified pool addresses its state regions in lane-row granules
    # (`_GranuleStateRef._granule_width`), which a wider value head would
    # straddle.
    if is_shared and d_v != num_lanes:
        raise NotImplementedError(
            f"The unified KV pool requires d_v == {num_lanes}; got d_v={d_v}."
        )
    # A bf16 recurrent state is packed (d_k // 2, 2, d_v) before the uint32
    # bitcast; the full-row granule for d_v > num_lanes is only exercised for
    # fp32 states.
    if not is_shared and jnp.dtype(recurrent_dtype) == jnp.bfloat16 and d_v > num_lanes:
        raise NotImplementedError(
            f"A bf16 recurrent state requires d_v <= {num_lanes}; got d_v={d_v}."
        )

    has_64m_vmem = (
        tpu_info.vmem_capacity_bytes >= config.GDNConfig.CALIBRATED_VMEM_CAPACITY_BYTES
    )
    head_groups = compute_gdn.select_head_groups(n_kq, n_v)
    conv_n_block = 512
    eligible_bf16_qkv_base = (
        has_64m_vmem
        and qkv_input_dtype == jnp.bfloat16
        and jnp.dtype(kernel_act_dtype) == jnp.float32
        and batch_size >= 256
        and batch_size % 16 == 0
        and num_spec_tokens == 0
        and not is_kda
        and kernel_size == 4
        and d_k == 128
        and d_v == 128
        and dim % conv_n_block == 0
        and (n_kq * d_k) % conv_n_block == 0
        and ((n_v // head_groups) * d_v) % conv_n_block == 0
    )

    act_in_dtype = (
        jnp.bfloat16.dtype if eligible_bf16_qkv_base else jnp.dtype(kernel_act_dtype)
    )
    packing = 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    decode_tile_size = min(decode_tile_size, batch_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes
    window_size = num_spec_tokens + 1
    decode_vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(window_size)

    if envs.TPU_ENABLE_GDN_DYNAMIC_TILING:
        decode_tile_size, resolved_mixed_tile_size = tiling.get_tile_sizes(
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
            decode_vmem_limit_bytes=config.GDNConfig.get_tiling_vmem_limit_bytes(
                window_size
            ),
            mixed_vmem_limit_bytes=config.GDNConfig.get_tiling_vmem_limit_bytes(),
            window_size=window_size,
            attention_mode=attention_mode,
            num_sublanes=num_sublanes,
        )
        logger.info_once(
            "%s dynamic tiling: num_seqs=%d batch=%d window=%d -> T=%d C=%d",
            attention_mode.value.upper(),
            num_seqs,
            batch_size,
            window_size,
            decode_tile_size,
            resolved_mixed_tile_size,
        )
    else:
        default_mixed = 128 if eligible_bf16_qkv_base else 64
        resolved_mixed_tile_size = (
            default_mixed if mixed_tile_size is None else mixed_tile_size
        )
        # Round the static clip down to a power of two so a small
        # non-power-of-two token bucket (e.g. TPU_TOKEN_BUCKET_EXTRA or tiny
        # max_num_batched_tokens) stays divisible by triangular_block_size.
        resolved_mixed_tile_size = tiling.clip_mixed_tile_size(
            resolved_mixed_tile_size, batch_size
        )
        # Each sequence holds 1 input state slot plus `window` output
        # checkpoint slots per buffer in VMEM, at every window size. Shrink
        # the tile so the double-buffered windows fit in roughly half the
        # scoped-VMEM budget.
        window = num_spec_tokens + 1
        num_buffers = config.GDNConfig.__dataclass_fields__["num_buffers"].default
        state_positions = 1 + window
        recurrent_bytes_per_seq = (
            state_positions * n_v * d_k * d_v * jnp.dtype(recurrent_dtype).itemsize
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
            # Static fallback keeps one sequence per KDA verify tile when
            # TPU_ENABLE_GDN_DYNAMIC_TILING is off.
            decode_tile_size = 1

    eligible_bf16_qkv = (
        eligible_bf16_qkv_base
        and resolved_mixed_tile_size >= 16
        and resolved_mixed_tile_size % 16 == 0
    )
    if not eligible_bf16_qkv:
        qkv = qkv.astype(kernel_act_dtype)
        act_in_dtype = qkv.dtype
        assert a.dtype == b.dtype == qkv.dtype == act_in_dtype
        packing = 4 // act_in_dtype.itemsize
        padded_batch_size = pl.cdiv(batch_size, packing) * packing

    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    # KDA's gate carries d_k channels per value head, so it pads out to a
    # lane multiple of n_v * d_k rather than of n_v.
    gate_dim = n_v * d_k if is_kda else n_v
    gate_padding_size = (pl.cdiv(gate_dim, num_lanes) * num_lanes) - gate_dim
    if eligible_bf16_qkv:
        qkv = qkv.reshape(padded_batch_size // 16, 16, dim)
    else:
        qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
        qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, gate_padding_size)))

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
            tile_size = resolved_mixed_tile_size
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
            beta_scale=beta_scale,
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

        metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)
        state_spec = jax.tree.map(lambda _: hbm_spec, in_states)
        scratch_shapes = cfg.get_scratch_shape_dict()

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

        if eligible_bf16_qkv:
            kernel_fn = functools.partial(
                outer_kernel_bf16,
                cfg=cfg,
                head_groups=head_groups,
                conv_n_block=conv_n_block,
            )
            if mode == config.GDNMode.PER_SEQ:
                scratch_shapes = {
                    "carry_conv_scratch_ref": pltpu.VMEM((16, dim), jnp.bfloat16),
                    "carry_recurrent_scratch_ref": pltpu.VMEM(
                        (1, n_v, d_k, d_v), jnp.float32
                    ),
                }
                vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(
                    window_size,
                    vmem_fraction=config.GDNConfig.FAST_GDN_VMEM_FRACTION,
                )
            else:
                vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(window_size)
        else:
            kernel_fn = functools.partial(outer_kernel, cfg=cfg)
            vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(window_size)

        return pl.pallas_call(
            kernel_fn,
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
