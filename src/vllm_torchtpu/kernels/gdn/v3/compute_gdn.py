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

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import config


LOG2E = 1.4426950408889634


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2-normalize along the last dim, accumulating in float32."""
    x_f32 = x.astype(jnp.float32)
    inv_norm = jax.lax.rsqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 * inv_norm).astype(x.dtype)


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32
        case 2:
            return jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


# NOTE: Fork of recurrent_scan_v2.py but applied various optimizations.
def invert_triangular_matrix(t: jax.Array, block_size=16) -> jax.Array:
    """Compute invert matrix of a given triauglar matrix."""

    # NOTE: if chunk_size=1, compiler will perform DCE.
    out_dtype = t.dtype
    chunk = t.shape[-1]
    block_size = min(block_size, chunk)
    # The blocked solve covers exactly num_blocks * block_size rows. A
    # remainder would be dropped and the output would no longer be
    # (..., chunk, chunk), which downstream surfaces as an unrelated shape
    # error (e.g. "array split does not result in an equal division").
    assert chunk % block_size == 0, (
        f"chunk ({chunk}) must be divisible by block_size ({block_size})"
    )
    num_blocks = chunk // block_size

    def local_forward_sub(t_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_t = t_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_t[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    iota_r = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
    iota_c = jax.lax.broadcasted_iota(jnp.int32, t.shape, 2)
    identity_mask = jnp.where(iota_r == iota_c, 1.0, 0.0)
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = identity_mask[:, start:end, :]

        if i == 0:
            target_b = e_block
        else:
            interaction_t = t[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jax.lax.dot(
                interaction_t,
                solved_x,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            target_b = e_block - prev_sum

        # NOTE: Utilize fp32 to minimize cost of sublane rolling.
        local_t = t[:, start:end, start:end].astype(jnp.float32)
        x_block = local_forward_sub(local_t, target_b)
        x_blocks.append(x_block.astype(out_dtype))

    return jnp.concatenate(x_blocks, axis=1)


def select_head_groups(num_kq_heads: int, num_v_heads: int) -> int:
    """Choose head groups so each group has <= 12 value heads when divisible."""
    target = (num_v_heads + 11) // 12
    if target <= 1:
        return 1
    for g in range(target, 1, -1):
        if num_v_heads % g == 0 and num_kq_heads % g == 0:
            return g
    return 1


def tri_inv_xor_dnc(
    L: jax.Array,
    *,
    use_fp32: bool = False,
) -> jax.Array:
    """Invert unit lower-triangular I + L via O(log2(BT)) batched MXU block-doubling."""
    BT = L.shape[-1]
    assert BT >= 4 and (BT & (BT - 1)) == 0, f"power-of-2 BT >= 4 required, got {BT}"

    r = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
    c = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
    strict = r > c
    diag = r == c
    zero = jnp.float32(0.0)
    one = jnp.float32(1.0)

    stage_prec = jax.lax.Precision.HIGHEST if use_fp32 else jax.lax.Precision.DEFAULT
    dims = (((2,), (1,)), ((0,), (0,)))

    def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            a, b, dims, precision=stage_prec, preferred_element_type=jnp.float32
        )

    L_strict = jnp.where(strict[None], L.astype(jnp.float32), zero)

    # Stage s = 1: exact 2x2 diagonal block inverses with 0 matmuls.
    mask_1 = (r == c + 1) & ((c & 1) == 0)
    T = jnp.where(diag[None], one, jnp.where(mask_1[None], -L_strict, zero))

    # Intermediate stages: s = 2, 4, ..., BT // 4.
    s = 2
    half_bt = BT // 2
    while s < half_bt:
        shift = (2 * s).bit_length() - 1
        mask_s = ((r >> shift) == (c >> shift)) & ((r & s) != 0) & ((c & s) == 0)
        B_s = jnp.where(mask_s[None], L_strict, zero)
        T = T - _dot(_dot(T, B_s), T)
        s *= 2

    # Final stage s = BT // 2: slice only sublane axis 1 (8-sublane aligned).
    mask_last = (r >= half_bt) & (c < half_bt)
    B_last = jnp.where(mask_last[None], L_strict, zero)
    tb_bot = _dot(T[:, half_bt:, :], B_last)
    upd_bot = _dot(tb_bot, T)
    T = jnp.concatenate([T[:, :half_bt, :], T[:, half_bt:, :] - upd_bot], axis=1)

    return jnp.where(strict[None], T, jnp.where(diag[None], one, zero))


def fused_transpose_broadcast(
    x: jax.Array, src_dim: int, dst_dim: int, dst_size: int | None = None
) -> jax.Array:
    """Perform 1D transpose where results are broadcasted along src_dim."""
    assert x.shape[dst_dim] == 1

    dtype = x.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(x.shape)
    mask_size = mask_shape[src_dim] if dst_size is None else dst_size
    mask_shape[dst_dim] = mask_size
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


def chunked_gdn_group(
    q_g: jax.Array,  # [n_kq_g, chunk, d_k] (L2-normed & scaled)
    k_g: jax.Array,  # [n_kq_g, chunk, d_k] (L2-normed)
    v_g: jax.Array,  # [n_v_g, chunk, d_v]
    g_cs_g: jax.Array,  # [n_v_g, chunk, 1] (log2 cumsum)
    beta_g: jax.Array,  # [n_v_g, chunk, 1]
    state_prev_g: jax.Array,  # [n_v_g, d_k, d_v]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Run one pure-SSA head group of chunked GDN using MXU XOR-DnC + exp2."""
    chunk = cfg.chunk_size
    n_v_g = v_g.shape[0]
    can_dnc = chunk >= 4 and (chunk & (chunk - 1)) == 0
    mask_dtype = get_mask_dtype(cfg.dtypes.compute)

    if chunk >= 8 and chunk % 8 == 0:
        g_mat = jnp.broadcast_to(g_cs_g, (n_v_g, chunk, chunk))
        g_diff_raw = g_mat - jnp.swapaxes(g_mat, 1, 2)
    else:
        g_cs_t = fused_transpose_broadcast(g_cs_g, src_dim=1, dst_dim=2)
        g_diff_raw = g_cs_g - g_cs_t

    iota_r = jax.lax.broadcasted_iota(mask_dtype, (n_v_g, chunk, chunk), 1)
    iota_c = jax.lax.broadcasted_iota(mask_dtype, (n_v_g, chunk, chunk), 2)
    identity_mask = iota_r == iota_c
    strictly_lower_mask = iota_r > iota_c
    lower_mask = iota_r >= iota_c

    # Mask upper-triangle BEFORE exp2 to prevent inf -> NaN at C=128.
    gating_map = jnp.exp2(jnp.where(lower_mask, g_diff_raw, 0.0))
    gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0.0)
    gating_backward = jnp.exp2(g_cs_g[:, -1:] - g_cs_g)
    gating_forward = jnp.exp2(g_cs_g)
    gating_last = gating_forward[:, -1:]

    # Single n_kq_g-head [k; q] @ k^T dot before GQA repeat.
    qk_g = jnp.concat([k_g, q_g], axis=1)
    kk_qk = jax.lax.dot(
        qk_g,
        k_g,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    k_k_t_g, q_k_t_g = jnp.split(kk_qk, 2, axis=1)
    k_k_t = jnp.repeat(k_k_t_g, cfg.v_per_kq_head, axis=0)
    gating_beta_k_k_t = (gating_map_masked * beta_g) * k_k_t

    if can_dnc:
        t_inv = tri_inv_xor_dnc(gating_beta_k_k_t, use_fp32=False).astype(
            cfg.dtypes.compute
        )
    else:
        t = jnp.where(identity_mask, 1.0, gating_beta_k_k_t)
        t_inv = invert_triangular_matrix(t, block_size=cfg.triangular_block_size)

    v_beta_g = v_g * beta_g
    k_rep = jnp.repeat(k_g, cfg.v_per_kq_head, axis=0)
    q_rep = jnp.repeat(q_g, cfg.v_per_kq_head, axis=0)
    k_beta_gating = k_rep * (beta_g * gating_forward)
    merged_v_k = jnp.concat([v_beta_g, k_beta_gating], axis=-1)
    merged_uw = jax.lax.dot(
        t_inv,
        merged_v_k,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    u, w = jnp.split(merged_uw, [cfg.v_head_dim], axis=-1)

    q_large_gating = q_rep * gating_forward
    merged_w_q = jnp.concat([w, q_large_gating], axis=1)
    merged_ws_out_updated = jax.lax.dot(
        merged_w_q,
        state_prev_g,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
    ws = ws.astype(cfg.dtypes.compute)
    u_ws = u - ws

    k_repeat_gating = k_rep * gating_backward
    state_new = jax.lax.dot(
        k_repeat_gating,
        u_ws,
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    state = state_prev_g * gating_last + state_new

    out_qk = jnp.repeat(q_k_t_g, cfg.v_per_kq_head, axis=0) * gating_map
    out_qk = jnp.where(lower_mask, out_qk, 0.0)
    out_new = jax.lax.dot(
        out_qk,
        u_ws,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    return out_updated + out_new, state


def chunked_gdn_per_seq(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array,  # [1, chunk, aligned_num_v_heads]
    beta: jax.Array,  # [1, chunk, aligned_num_v_heads]
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform chunked GDN over input [num_heads, chunk, head_dim]."""
    chunk = cfg.chunk_size
    g_log2_2d = (gating_log[0] * jnp.float32(LOG2E)).astype(jnp.float32)
    r_c = jax.lax.broadcasted_iota(jnp.int32, (chunk, chunk), 0)
    c_c = jax.lax.broadcasted_iota(jnp.int32, (chunk, chunk), 1)
    tril_f32 = (r_c >= c_c).astype(jnp.float32)
    g_cs_2d = jax.lax.dot_general(
        tril_f32,
        g_log2_2d,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    g_cs_all = fused_transpose_broadcast(
        g_cs_2d[None, :, :], src_dim=2, dst_dim=0, dst_size=cfg.num_v_heads
    )
    beta_all = fused_transpose_broadcast(
        beta, src_dim=2, dst_dim=0, dst_size=cfg.num_v_heads
    )

    head_groups = select_head_groups(cfg.num_kq_heads, cfg.num_v_heads)
    if head_groups > 1:
        kq_per = cfg.num_kq_heads // head_groups
        v_per = cfg.num_v_heads // head_groups
        out_parts = []
        state_parts = []
        for g in range(head_groups):
            kq0, kq1 = g * kq_per, (g + 1) * kq_per
            v0, v1 = g * v_per, (g + 1) * v_per
            og, sg = chunked_gdn_group(
                q_large[kq0:kq1],
                k_large[kq0:kq1],
                v_large[v0:v1],
                g_cs_all[v0:v1],
                beta_all[v0:v1],
                state_prev[v0:v1],
                cfg=cfg,
            )
            out_parts.append(og)
            state_parts.append(sg)
        return jnp.concat(out_parts, axis=0), jnp.concat(state_parts, axis=0)

    return chunked_gdn_group(
        q_large, k_large, v_large, g_cs_all, beta_all, state_prev, cfg=cfg
    )


def chunked_gdn(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform chunked GDN over input [seq, num_heads, chunk, head_dim].

    The returned state is [seq, 1, num_v_heads, kq_head_dim, v_head_dim]: the
    chunked path only produces the final state, so it is never taken when
    more than one checkpoint per sequence is required.
    """

    assert not cfg.use_recurrent

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1), 2
    )
    mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

    # [seqs, num_kq_heads, chunk, kq_head_dim]
    q_large = jnp.where(mask, q_large.astype(cfg.dtypes.compute), 0)
    k_large = jnp.where(mask, k_large.astype(cfg.dtypes.compute), 0)
    # [seqs, num_v_heads, chunk, v_head_dim]
    v_large = jnp.where(mask, v_large.astype(cfg.dtypes.compute), 0)

    b_large = b_large.astype(cfg.dtypes.compute)
    a_large = a_large.astype(cfg.dtypes.compute)

    a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)

    # NOTE: Any element-wise computations should occur before repeat.
    q_large = l2_norm(q_large)
    q_scale = cfg.kq_head_dim**-0.5
    q_large *= q_scale
    k_large = l2_norm(k_large)

    # [seqs, 1, chunk, num_v_heads]
    beta = jax.nn.sigmoid(b_large)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_large + dt_bias)

    beta = jnp.where(mask, beta, 0)
    # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
    # be masked to 1 since it signifies that strength of state from previous row
    # will be 1 (i.e., no decay) if current row is invalid.
    gating_log = jnp.where(mask, gating_log, 0)

    out_list = []
    state_list = []
    for idx in range(cfg.seq_tile_size):
        out, state = chunked_gdn_per_seq(
            q_large[idx],
            k_large[idx],
            v_large[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfg,
        )
        out_list.append(out.swapaxes(0, 1))
        state_list.append(state)
    out = jnp.stack(out_list, axis=0)
    # Single checkpoint per sequence, laid out like the recurrent path.
    state = jnp.stack(state_list, axis=0)[:, jnp.newaxis]
    return out, state


def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform recurrent GDN over input [seq, num_heads, chunk, 1, head_dim].

    The returned state has one checkpoint per window position: [seq,
    window_size, num_v_heads, kq_head_dim, v_head_dim]. Positions >=
    real_sizes repeat the last valid state; they are never written back to
    HBM.
    """

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2
    )
    mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

    a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    q_scale = cfg.kq_head_dim**-0.5

    # Flatten `(seq_tile_size, num_v_heads)` into one leading batch axis for
    # rank-3 `[B * n_v, 2, d_k] x [B * n_v, d_k, d_v]` `jax.lax.dot_general`
    # with `dimension_numbers=(((2,), (1,)), ((0,), (0,)))`, fusing `[k_c, q_c]`
    # along the length-2 axis so one matmul computes both projections per step.
    #
    # Per-step shapes (B = seq_tile_size, n_kq = num_kq_heads, n_v = num_v_heads):
    #   q_compact[:, :, c_idx]: [B, n_kq, 1, d_k] -> q_c:         [B, n_kq, 1, d_k]
    #   k_compact[:, :, c_idx]: [B, n_kq, 1, d_k] -> k_c:         [B, n_kq, 1, d_k]
    #   v_compact[:, :, c_idx]: [B, n_v,  1, d_v] -> v_curr:      [B, n_v,  1, d_v]
    #   beta_curr, gating_curr: [B, n_v,  1, 1]
    #   state, state_updated:   [B, n_v,  d_k, d_v]
    #   kq_merged:              [B, n_v,  2, d_k] -> flat [B * n_v, 2, d_k]
    #   k_curr_t (transposed):  [B, n_v,  d_k, 1]
    out_list = []
    state_list = []
    state = state_prev
    for c_idx in range(cfg.chunk_size):
        mask_c = mask[:, :, c_idx]
        q_c = jnp.where(mask_c, q_compact[:, :, c_idx].astype(cfg.dtypes.compute), 0)
        k_c = jnp.where(mask_c, k_compact[:, :, c_idx].astype(cfg.dtypes.compute), 0)
        v_curr = jnp.where(mask_c, v_compact[:, :, c_idx].astype(cfg.dtypes.compute), 0)

        q_c = l2_norm(q_c) * q_scale
        k_c = l2_norm(k_c)

        b_c = b_compact[:, :, c_idx].astype(cfg.dtypes.compute)
        a_c = a_compact[:, :, c_idx].astype(cfg.dtypes.compute)
        beta_c = jnp.where(mask_c, jax.nn.sigmoid(b_c), 0)
        # NOTE: Masked gating_c will evaluate to jnp.exp(0)=1. gating (decay)
        # must be masked to 1 since it signifies that strength of state from
        # previous row will be 1 (i.e., no decay) if current row is invalid.
        gating_c = jnp.where(
            mask_c, -jnp.exp(a_log) * jax.nn.softplus(a_c + dt_bias), 0
        )
        gating_c = jnp.exp(gating_c)

        beta_curr = fused_transpose_broadcast(beta_c, src_dim=3, dst_dim=1)[
            :, : cfg.num_v_heads
        ]
        gating_curr = fused_transpose_broadcast(gating_c, src_dim=3, dst_dim=1)[
            :, : cfg.num_v_heads
        ]

        # 1. State decay update
        state_updated = state * gating_curr

        # 2. Fused [k, q] projection against state
        b_size, n_v_heads = cfg.seq_tile_size, cfg.num_v_heads
        kq_merged = jnp.repeat(
            jnp.concat([k_c, q_c], axis=2), cfg.v_per_kq_head, axis=1
        )  # [B, n_v, 2, d_k]
        kq_merged_flat = kq_merged.reshape(-1, 2, cfg.kq_head_dim)
        state_updated_flat = state_updated.reshape(-1, cfg.kq_head_dim, cfg.v_head_dim)
        kq_proj_flat = jax.lax.dot_general(
            kq_merged_flat,
            state_updated_flat,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).astype(cfg.dtypes.compute)  # [B * n_v, 2, d_v]
        kq_proj = kq_proj_flat.reshape(b_size, n_v_heads, 2, cfg.v_head_dim)
        v_updated = kq_proj[:, :, :1, :]
        out_decayed = kq_proj[:, :, 1:, :]

        # 3. Output difference and rank-1 correction (fp32 inner-product reduction)
        v_diff = v_curr - v_updated
        v_new = beta_curr * v_diff
        qk_dot = jnp.repeat(
            jnp.sum(
                q_c.astype(jnp.float32) * k_c.astype(jnp.float32),
                axis=-1,
                keepdims=True,
            ),
            cfg.v_per_kq_head,
            axis=1,
        )  # [B, n_v, 1, 1]
        out = (out_decayed + qk_dot * v_new).astype(cfg.dtypes.compute)

        # 4. State rank-1 update
        k_curr_t = jnp.repeat(
            fused_transpose_broadcast(k_c, src_dim=3, dst_dim=2),
            cfg.v_per_kq_head,
            axis=1,
        )  # [B, n_v, d_k, 1]
        # Multiplication with k_curr_t is deferred until after the scalar v_new
        # update because it expands the dimension size by kq_head_dim.
        state_new = k_curr_t * v_new  # [B, n_v, d_k, d_v]
        state = state_updated + state_new

        out_list.append(out[:, :, 0, :])
        # Rows past real_sizes are masked out above, so the state stops
        # changing there and trailing checkpoints repeat the state of the last
        # real token. Checkpoints of the positions dropped here are dead code
        # that the compiler eliminates.
        if c_idx >= cfg.chunk_size - cfg.window_size:
            state_list.append(state)

    return jnp.stack(out_list, axis=1), jnp.stack(state_list, axis=1)
