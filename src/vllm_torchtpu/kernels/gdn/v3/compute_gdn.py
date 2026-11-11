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
"""Chunked and recurrent GDN compute."""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import config


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2-normalize along the last dim, accumulating in float32."""
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / norm).astype(x.dtype)


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32
        case 2:
            return jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def solve_triangular_rhs(
    strict_lower: jax.Array, rhs: jax.Array, block_size: int
) -> jax.Array:
    """Solves the unit lower-triangular system (I + L) @ x = rhs in blocks.

    Args:
        strict_lower: L, the strictly lower-triangular part of the system
            matrix, [num_heads, chunk, chunk]. Its diagonal and upper triangle
            must be zero; the unit diagonal is implicit.
        rhs: Right-hand side [num_heads, chunk, n].
        block_size: Diagonal block size B. Blocks are solved in order; each
            block first removes the coupling to the already-solved rows with
            one matmul and then runs a dense in-block forward substitution.

    Returns:
        x with the dtype of `strict_lower`.
    """
    out_dtype = strict_lower.dtype
    chunk = strict_lower.shape[-1]
    block_size = min(block_size, chunk)
    num_blocks = (chunk + block_size - 1) // block_size
    rhs_f32 = rhs.astype(jnp.float32)

    def local_forward_sub(l_block: jax.Array, b_block: jax.Array) -> jax.Array:
        # Column-oriented (right-looking) substitution over the whole dense
        # [num_heads, B, n] block: after step j, row j is final and the rows
        # below it have column j's contribution removed. Rows <= j are
        # untouched because their coefficient in the strictly lower block is
        # zero, so each step is one multiply-subtract over B rows held densely
        # in sublanes. A row-by-row chain costs one vreg op per head per
        # (row, column) pair instead, which made the solve VALU-bound at 8-
        # and 16-row blocks (few V heads per core).
        x = b_block
        for j in range(l_block.shape[1] - 1):
            x = x - l_block[:, :, j : j + 1] * x[:, j : j + 1, :]
        return x

    x_blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, chunk)
        target_b = rhs_f32[:, start:end, :]
        if i > 0:
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jax.lax.dot(
                strict_lower[:, start:end, :start],
                solved_x,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            target_b = target_b - prev_sum

        l_block = strict_lower[:, start:end, start:end].astype(jnp.float32)
        x_blocks.append(local_forward_sub(l_block, target_b).astype(out_dtype))

    return jnp.concatenate(x_blocks, axis=1)


def fused_transpose_broadcast(x: jax.Array, src_dim: int, dst_dim: int) -> jax.Array:
    """Perform 1D transpose where results are broadcasted along src_dim."""
    assert x.shape[dst_dim] == 1

    dtype = x.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(x.shape)
    mask_size = mask_shape[src_dim]
    mask_shape[dst_dim] = mask_size
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


def _pack_v_head_pairs(x: jax.Array, v_per_kq_head: int) -> jax.Array:
    """Packs the two V-head halves of each KQ head group along the lane axis."""
    half_v = v_per_kq_head // 2
    group_starts = range(0, x.shape[0], v_per_kq_head)
    first_half = jnp.concat([x[s : s + half_v] for s in group_starts], axis=0)
    second_half = jnp.concat(
        [x[s + half_v : s + v_per_kq_head] for s in group_starts], axis=0
    )
    return jnp.concat([first_half, second_half], axis=-1)


def _unpack_v_head_pairs(x_pairs: jax.Array, v_per_kq_head: int) -> jax.Array:
    """Unpacks V-head pairs from the lane axis back to the head axis."""
    half_v = v_per_kq_head // 2
    first_half, second_half = jnp.split(x_pairs, 2, axis=-1)
    heads = []
    for s in range(0, x_pairs.shape[0], half_v):
        heads += [first_half[s : s + half_v], second_half[s : s + half_v]]
    return jnp.concat(heads, axis=0)


def _dot_kq_with_v_heads(
    kq_operand: jax.Array,  # [num_kq_heads, ...]
    v_operand: jax.Array,  # [num_v_heads, ..., v_head_dim]
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> jax.Array:
    """Contracts a per-KQ-head operand with a per-V-head operand.

    An even number of V heads per KQ head is packed in pairs along the lane
    axis, so that each matmul fills 2 * v_head_dim MXU columns.
    """
    v_per_kq_head = v_operand.shape[0] // kq_operand.shape[0]
    pair_v_heads = v_per_kq_head % 2 == 0
    kq_repeats = v_per_kq_head // 2 if pair_v_heads else v_per_kq_head
    kq_operand = jnp.repeat(kq_operand, kq_repeats, axis=0)
    if pair_v_heads:
        v_operand = _pack_v_head_pairs(v_operand, v_per_kq_head)
    out = jax.lax.dot(
        kq_operand,
        v_operand,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )
    if pair_v_heads:
        out = _unpack_v_head_pairs(out, v_per_kq_head)
    return out


def chunked_gdn_per_seq(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array,  # [1, 1, num_v_heads]
    beta: jax.Array,  # [1, 1, num_v_heads]
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform chunked GDN over input [num_heads, chunk, head_dim]."""

    # Compute cumulative sum of decay.
    # [1, 1, num_v_heads]
    g_cum_sum_list = [gating_log[:, :1]]
    for row in range(1, cfg.chunk_size):
        g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row : row + 1])
    # [1, chunk, num_v_heads]
    g_cum_sum_log = jnp.concat(g_cum_sum_list, axis=1)

    # [num_v_heads, chunk, 1]
    g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log, src_dim=2, dst_dim=0)
    g_cum_sum_log = g_cum_sum_log[: cfg.num_v_heads]
    beta = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)
    beta_large = beta[: cfg.num_v_heads]

    # [num_v_heads, 1, chunk]
    g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log, src_dim=1, dst_dim=2)
    beta_t = fused_transpose_broadcast(beta_large, src_dim=1, dst_dim=2)
    # [num_v_heads, chunk, chunk]
    g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
    gating_map = jnp.exp(g_cum_sum_diff_log)
    # [num_v_heads, chunk, 1]
    gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
    # [num_v_heads, chunk, 1]
    gating_forward = jnp.exp(g_cum_sum_log)
    # [num_v_heads, 1, 1]
    gating_last = gating_forward[:, -1:]

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota_r = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
    iota_c = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
    strictly_lower_mask = iota_r > iota_c
    lower_mask = iota_r >= iota_c
    # [num_v_heads, chunk, chunk]
    gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)

    # Contract K @ K^T once per KQ head before repeating across V heads.
    # [num_kq_heads, chunk, chunk]
    k_k_t = jax.lax.dot(
        k_large,
        k_large,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)

    # [num_v_heads, chunk, chunk]
    beta_k_k_t = jnp.repeat(k_k_t, cfg.v_per_kq_head, axis=0) * beta_t
    # Strictly lower-triangular part L of the column-scaled system T = I + L;
    # the solve below takes L directly, so T is never materialized.
    gating_beta_k_k_t = gating_map_masked * beta_k_k_t

    # State-first KQ-head-factored projection: contract unscaled [K; Q] against
    # state_prev before the triangular solve, so the solve has a single
    # 128-column right-hand side and W is never formed.
    # [num_kq_heads, 2 * chunk, kq_head_dim]
    merged_kq = jnp.concat([k_large, q_large], axis=1)
    # [num_v_heads, 2 * chunk, v_head_dim]
    kq_state = _dot_kq_with_v_heads(merged_kq, state_prev, (((2,), (1,)), ((0,), (0,))))

    # [num_v_heads, chunk, v_head_dim]
    k_state, q_state = jnp.split(kq_state, 2, axis=1)
    out_updated = q_state * gating_forward

    # Form unscaled innovation residual v_diff = V - gf * (K @ S_prev) in fp32,
    # solve the column-scaled system (I + L) @ x = v_diff directly without
    # materializing T^-1, and post-multiply by beta_large.
    # [num_v_heads, chunk, v_head_dim]
    v_diff = v_large - k_state * gating_forward
    u_ws = (
        solve_triangular_rhs(
            gating_beta_k_k_t, v_diff, block_size=cfg.triangular_block_size
        )
        * beta_large
    )

    # Absorb gating_backward into u_ws so the unscaled K (transposed once per KQ
    # head) is shared by all of its V heads.
    # [num_v_heads, chunk, v_head_dim]
    u_ws_gated = u_ws * gating_backward
    # [num_v_heads, kq_head_dim, v_head_dim]
    state_new = _dot_kq_with_v_heads(k_large, u_ws_gated, (((1,), (1,)), ((0,), (0,))))

    # [num_v_heads, kq_head_dim, v_head_dim]
    state_updated = state_prev * gating_last
    state = state_updated + state_new

    # [num_kq_heads, chunk, chunk]
    out_qk = jax.lax.dot(
        q_large,
        k_large,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    # NOTE: must perform repeat after matmul to reduce required compute.
    # [num_v_heads, chunk, chunk]
    out_qk = jnp.repeat(out_qk, cfg.v_per_kq_head, axis=0)
    out_qk *= gating_map
    out_qk = jnp.where(lower_mask, out_qk, 0)

    # [num_v_heads, chunk, v_head_dim]
    out_new = jax.lax.dot(
        out_qk,
        u_ws,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    out = out_updated + out_new

    return out, state


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


def recurrent_gdn_per_seq(
    q_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact_t: jax.Array,  # [num_kq_heads, chunk, kq_head_dim, 1]
    v_compact: jax.Array,  # [num_v_heads, chunk, 1, v_head_dim]
    gating_log: jax.Array,  # [num_v_heads, chunk, 1, 1]
    beta: jax.Array,  # [num_v_heads, chunk, 1, 1]
    state: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform recurrent GDN over input [num_heads, chunk, 1, head_dim].

    The returned state stacks the post-token state of the last
    `cfgs.window_size` window positions ([window_size, num_v_heads,
    kq_head_dim, v_head_dim]); with `window_size == 1` that is just the final
    state.
    """

    out_list = []
    state_list = []
    for c_idx in range(cfgs.chunk_size):
        # [num_v_heads, 1, kq_head_dim]
        q_curr = q_compact[:, c_idx]
        q_curr = jnp.repeat(q_curr, cfgs.v_per_kq_head, axis=0)
        k_curr = k_compact[:, c_idx]
        k_curr = jnp.repeat(k_curr, cfgs.v_per_kq_head, axis=0)

        # [num_v_heads, 1, v_head_dim]
        v_curr = v_compact[:, c_idx]

        # [num_v_heads, kq_head_dim, 1]
        k_curr_t = k_compact_t[:, c_idx]
        k_curr_t = jnp.repeat(k_curr_t, cfgs.v_per_kq_head, axis=0)

        # [num_v_heads, 1, 1]
        beta_curr = beta[:, c_idx]
        gating_curr = gating_log[:, c_idx]

        # [num_v_heads, kq_head_dim, v_head_dim]
        state_updated = state * gating_curr

        # [num_v_heads, 1, v_head_dim]
        v_updated = jax.lax.dot(
            k_curr,
            state_updated,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)

        # [num_v_heads, 1, v_head_dim]
        v_diff = v_curr - v_updated
        v_new = beta_curr * v_diff

        # [num_v_heads, kq_head_dim, v_head_dim]
        # NOTE: Multiplication with k_curr_t needs to be deferred as much as
        # possible as it expands the dimension size by kq_head_dim.
        state_new = k_curr_t * v_new
        # [num_v_heads, kq_head_dim, v_head_dim]
        state = state_updated + state_new

        # [num_v_heads, 1, v_head_dim]
        out = jax.lax.dot(
            q_curr,
            state,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)

        out_list.append(out[:, 0, :])
        # NOTE: Rows past real_sizes are masked out by the caller, so the state
        # stops changing there and trailing checkpoints simply repeat the state
        # of the last real token. Checkpoints of the positions dropped here are
        # dead code the compiler eliminates.
        if c_idx >= cfgs.chunk_size - cfgs.window_size:
            state_list.append(state)

    return jnp.stack(out_list, axis=0), jnp.stack(state_list, axis=0)


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

    # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
    q_compact = jnp.where(mask, q_compact.astype(cfg.dtypes.compute), 0)
    k_compact = jnp.where(mask, k_compact.astype(cfg.dtypes.compute), 0)
    # [seqs, num_v_heads, chunk, 1, v_head_dim]
    v_compact = jnp.where(mask, v_compact.astype(cfg.dtypes.compute), 0)

    b_compact = b_compact.astype(cfg.dtypes.compute)
    a_compact = a_compact.astype(cfg.dtypes.compute)

    a_log = a_log.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)

    # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
    q_compact = l2_norm(q_compact)
    q_scale = cfg.kq_head_dim**-0.5
    q_compact *= q_scale
    k_compact = l2_norm(k_compact)
    k_compact_t = fused_transpose_broadcast(k_compact, src_dim=4, dst_dim=3)

    beta = jax.nn.sigmoid(b_compact)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_compact + dt_bias)

    beta = jnp.where(mask, beta, 0)
    # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
    # be masked to 1 since it signifies that strength of state from previous row
    # will be 1 (i.e., no decay) if current row is invalid.
    gating_log = jnp.where(mask, gating_log, 0)
    gating_log = jnp.exp(gating_log)

    beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
    beta = beta[:, : cfg.num_v_heads]
    gating_log = fused_transpose_broadcast(gating_log, src_dim=4, dst_dim=1)
    gating_log = gating_log[:, : cfg.num_v_heads]

    if cfg.chunk_size == 1 and cfg.window_size == 1:
        # Optimized decode path: fused [k, q] projection against state.
        # Shapes:
        #   q_compact:    [B, n_kq, 1, 1, d_k] -> q_curr:      [B, n_v, 1, d_k]
        #   k_compact:    [B, n_kq, 1, 1, d_k] -> k_curr:      [B, n_v, 1, d_k]
        #   k_compact_t:  [B, n_kq, 1, d_k, 1] -> k_curr_t:    [B, n_v, d_k, 1]
        #   v_compact:    [B, n_v, 1, 1, d_v]  -> v_curr:      [B, n_v, 1, d_v]
        #   beta:         [B, n_v, 1, 1, 1]    -> beta_curr:   [B, n_v, 1, 1]
        #   gating_log:   [B, n_v, 1, 1, 1]    -> gating_curr: [B, n_v, 1, 1]
        q_curr = jnp.repeat(q_compact[:, :, 0, 0, :], cfg.v_per_kq_head, axis=1)
        q_curr = jnp.expand_dims(q_curr, axis=2)

        k_curr = jnp.repeat(k_compact[:, :, 0, 0, :], cfg.v_per_kq_head, axis=1)
        k_curr = jnp.expand_dims(k_curr, axis=2)

        k_curr_t = jnp.repeat(k_compact_t[:, :, 0, :, 0], cfg.v_per_kq_head, axis=1)
        k_curr_t = jnp.expand_dims(k_curr_t, axis=3)

        v_curr = jnp.expand_dims(v_compact[:, :, 0, 0, :], axis=2)
        beta_curr = beta[:, :, 0, 0, 0].reshape(
            cfg.seq_tile_size, cfg.num_v_heads, 1, 1
        )
        gating_curr = gating_log[:, :, 0, 0, 0].reshape(
            cfg.seq_tile_size, cfg.num_v_heads, 1, 1
        )

        # 1. State decay update
        state_updated = state_prev * gating_curr

        # 2. Fused [k, q] projection against state
        b_size, n_v_heads = cfg.seq_tile_size, cfg.num_v_heads
        kq_merged = jnp.concat([k_curr, q_curr], axis=2)  # [B, n_v, 2, d_k]
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

        # 3. Output difference
        v_diff = v_curr - v_updated
        v_new = beta_curr * v_diff

        qk_dot = jnp.sum(q_curr * k_curr, axis=-1, keepdims=True)  # [B, n_v, 1, 1]
        out = out_decayed + qk_dot * v_new  # [B, n_v, 1, d_v]

        # 4. State rank-1 update
        state_new = k_curr_t * v_new  # [B, n_v, d_k, d_v]
        new_recurrent_state = state_updated + state_new

        # Match Pallas output shape [B, 1, n_v, d_v]:
        out = out.swapaxes(1, 2)
        # Output single checkpoint per sequence, matching [seq, window_size, ...]
        return out, jnp.expand_dims(new_recurrent_state, axis=1)

    out_list = []
    new_state_list = []

    for idx in range(cfg.seq_tile_size):
        out, state = recurrent_gdn_per_seq(
            q_compact[idx],
            k_compact[idx],
            k_compact_t[idx],
            v_compact[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfg,
        )
        out_list.append(out)
        new_state_list.append(state)

    out = jnp.stack(out_list, axis=0)
    new_recurrent_state = jnp.stack(new_state_list, axis=0)

    return out, new_recurrent_state
