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

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import config


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
    return x / norm


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


def solve_triangular_rhs(
    strict_lower: jax.Array, rhs: jax.Array, block_size: int
) -> jax.Array:
    """Solves the unit lower-triangular system (I + L) @ x = rhs in blocks.

    Ported from Caslyn Tonelli's GDN v3 direct-solve stack (`fab1042538`).

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
        # in sublanes.
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

    When the number of V heads per KQ head is even and v_head_dim % 128 == 0,
    V heads are packed in pairs along the lane axis so each matmul fills
    2 * v_head_dim MXU columns; otherwise V heads are contracted individually.
    """
    v_per_kq_head = v_operand.shape[0] // kq_operand.shape[0]
    pair_v_heads = v_per_kq_head % 2 == 0 and v_operand.shape[-1] % 128 == 0
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


def _chunk_intra_direct(
    q_large: jax.Array,  # [heads_kq, chunk, kq_head_dim]
    k_large: jax.Array,  # [heads_kq, chunk, kq_head_dim]
    g_cum_sum_log: jax.Array,  # [heads_v, chunk, 1]
    g_cum_sum_log_t: jax.Array,  # [heads_v, 1, chunk]
    beta_t: jax.Array,  # [heads_v, 1, chunk]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, ...]:
    """State-independent intra-chunk prep for the direct-solve GDN path.

    Factorizes K @ K^T once per KQ head group (`ed54e5b64e`) and omits the
    triangular inverse buffer `t_inv` (`fab1042538`).
    """
    # [heads_v, chunk, chunk]
    g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
    gating_map = jnp.exp(g_cum_sum_diff_log)
    # [heads_v, chunk, 1]
    gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
    # [heads_v, chunk, 1]
    gating_forward = jnp.exp(g_cum_sum_log)
    # [heads_v, 1, 1]
    gating_last = gating_forward[:, -1:]

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota_r = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
    iota_c = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
    strictly_lower_mask = iota_r > iota_c
    lower_mask = iota_r >= iota_c
    # [heads_v, chunk, chunk]
    gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)

    # Contract K @ K^T once per KQ head before repeating across V heads.
    # [heads_kq, chunk, chunk]
    k_k_t = jax.lax.dot(
        k_large,
        k_large,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)

    # Strictly lower-triangular part L of the column-scaled system T = I + L.
    # [heads_v, chunk, chunk]
    beta_k_k_t = jnp.repeat(k_k_t, cfg.v_per_kq_head, axis=0) * beta_t
    gating_beta_k_k_t = gating_map_masked * beta_k_k_t

    # [heads_kq, chunk, chunk]
    out_qk = jax.lax.dot(
        q_large,
        k_large,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    # NOTE: must perform repeat after matmul to reduce required compute.
    # [heads_v, chunk, chunk]
    out_qk = jnp.repeat(out_qk, cfg.v_per_kq_head, axis=0)
    out_qk *= gating_map
    out_qk = jnp.where(lower_mask, out_qk, 0)

    return gating_beta_k_k_t, gating_forward, gating_backward, gating_last, out_qk


def _chunk_inter_direct(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    beta_large: jax.Array,  # [num_v_heads, chunk, 1]
    gating_beta_k_k_t: jax.Array,  # [num_v_heads, chunk, chunk]
    gating_forward: jax.Array,  # [num_v_heads, chunk, 1]
    gating_backward: jax.Array,  # [num_v_heads, chunk, 1]
    gating_last: jax.Array,  # [num_v_heads, 1, 1]
    out_qk: jax.Array,  # [num_v_heads, chunk, chunk]
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """State-dependent part of one chunk using direct triangular solve."""
    # Project unscaled [K; Q] against state_prev before solve (`bbaac5dd58`).
    # [num_kq_heads, 2 * chunk, kq_head_dim]
    merged_kq = jnp.concat([k_large, q_large], axis=1)
    # [num_v_heads, 2 * chunk, v_head_dim]
    kq_state = _dot_kq_with_v_heads(merged_kq, state_prev, (((2,), (1,)), ((0,), (0,))))
    # [num_v_heads, chunk, v_head_dim]
    k_state, q_state = jnp.split(kq_state, 2, axis=1)
    out_updated = q_state * gating_forward

    # Solve (I + L) @ x = v_diff directly in fp32 without T^-1 (`fab1042538`).
    v_diff = v_large - k_state * gating_forward
    u_ws = (
        solve_triangular_rhs(
            gating_beta_k_k_t, v_diff, block_size=cfg.triangular_block_size
        )
        * beta_large
    )

    # Absorb gating_backward into u_ws so sibling V heads share unscaled K.
    u_ws_gated = u_ws * gating_backward
    # [num_v_heads, kq_head_dim, v_head_dim]
    state_new = _dot_kq_with_v_heads(k_large, u_ws_gated, (((1,), (1,)), ((0,), (0,))))
    state_updated = state_prev * gating_last
    state = state_updated + state_new

    # [num_v_heads, chunk, v_head_dim]
    out_new = jax.lax.dot(
        out_qk,
        u_ws,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    out = out_updated + out_new
    return out, state


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

    # NOTE: Repeat along non lane/sublane dim is free.
    q_repeat = jnp.repeat(q_large, cfg.v_per_kq_head, axis=0)
    k_repeat = jnp.repeat(k_large, cfg.v_per_kq_head, axis=0)

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
    if envs.VLLM_TPU_GDN_DIRECT_SOLVE:
        beta_t = fused_transpose_broadcast(beta_large, src_dim=1, dst_dim=2)
        parts = _chunk_intra_direct(
            q_large, k_large, g_cum_sum_log, g_cum_sum_log_t, beta_t, cfg
        )
        return _chunk_inter_direct(
            q_large, k_large, v_large, beta_large, *parts, state_prev, cfg
        )
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
    identity_mask = iota_r == iota_c
    strictly_lower_mask = iota_r > iota_c
    lower_mask = iota_r >= iota_c
    # [num_v_heads, chunk, chunk]
    gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)

    # [num_v_heads, chunk, kq_head_dim]
    k_beta_repeat = k_repeat * beta_large

    # [num_v_heads, chunk, chunk]
    beta_k_k_t = jax.lax.dot(
        k_beta_repeat,
        k_repeat,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    gating_beta_k_k_t = gating_map_masked * beta_k_k_t
    t = jnp.where(identity_mask, 1, gating_beta_k_k_t)

    # [num_v_heads, chunk, chunk]
    t_inv = invert_triangular_matrix(t, block_size=cfg.triangular_block_size)

    # [num_v_heads, chunk, v_head_dim]
    v_beta_large = v_large * beta_large
    # [num_v_heads, chunk, kv_head_dim]
    k_beta_gating = k_beta_repeat * gating_forward
    # NOTE: If v_head_dim < mxu size, concatenating them will help increase mxu
    # utilization. Also, if v_head_dim is multiple of lane size, concat / split
    # along lane dim is free - making this optimization strictly beneficial.
    # [num_v_heads, chunk, v_head_dim + kq_head_dim]
    merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
    merged_uw = jax.lax.dot(
        t_inv,
        merged_v_k,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)

    # [num_v_heads, chunk, v_head_dim]
    u, w = jnp.split(merged_uw, [cfg.v_head_dim], axis=-1)

    # [num_v_heads, chunk, kq_head_dim]
    q_large_gating = q_repeat * gating_forward
    # NOTE: Concatenate lhs with same rhs to leverage weight
    # stationary architecture.
    # [num_v_heads, 2 * chunk, kq_head_dim]
    merged_w_q = jnp.concat([w, q_large_gating], axis=1)
    # [num_v_heads, 2 * chunk, v_head_dim]
    merged_ws_out_updated = jax.lax.dot(
        merged_w_q,
        state_prev,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    # NOTE: Splitting along non sublane/lane dim is free.
    ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
    ws = ws.astype(cfg.dtypes.compute)

    # [num_v_heads, chunk, v_head_dim]
    u_ws = u - ws

    # [num_v_heads, chunk, kq_head_dim]
    k_repeat_gating = k_repeat * gating_backward

    # [num_v_heads, kq_head_dim, v_head_dim]
    state_new = jax.lax.dot(
        k_repeat_gating,
        u_ws,
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

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
    state. In LOG mode it is the state after position 0 alone
    ([1, ...]), and a third result stacks the update vectors v_new of
    positions 1.. ([chunk - 1, num_v_heads, 1, v_head_dim]) for the log.
    """

    out_list = []
    state_list = []
    v_new_list = []
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
        if cfgs.writes_window_log:
            if c_idx == 0:
                state_list.append(state)
            else:
                v_new_list.append(v_new)
        elif c_idx >= cfgs.chunk_size - cfgs.window_size:
            state_list.append(state)

    if cfgs.writes_window_log:
        return (
            jnp.stack(out_list, axis=0),
            jnp.stack(state_list, axis=0),
            jnp.stack(v_new_list, axis=0),
        )
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

    In LOG mode (BATCHED, chunk == window) the state is the one after window
    position 0 ([seq, 1, ...]) and a third result holds the draft log of
    positions 1..K: gates [seq, K, aligned_num_v_heads] (the decay after
    exp, before the lane-to-sublane transpose), keys [seq, K, num_kq_heads,
    1, kq_head_dim] (after l2_norm) and updates [seq, K, num_v_heads, 1,
    v_head_dim] (v_new). Masked positions log g = 1, k = 0 and v_new = 0.
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

    # [seqs, chunk, aligned_num_v_heads]: the gates as the log stores them.
    log_gates = gating_log[:, 0, :, 0, :]

    beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
    beta = beta[:, : cfg.num_v_heads]
    gating_log = fused_transpose_broadcast(gating_log, src_dim=4, dst_dim=1)
    gating_log = gating_log[:, : cfg.num_v_heads]

    out_list = []
    new_state_list = []
    v_new_list = []

    for idx in range(cfg.seq_tile_size):
        out, state, *v_new = recurrent_gdn_per_seq(
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
        v_new_list.extend(v_new)

    out = jnp.stack(out_list, axis=0)
    new_recurrent_state = jnp.stack(new_state_list, axis=0)

    if cfg.writes_window_log:
        assert cfg.chunk_size == cfg.log_positions + 1, (cfg.chunk_size, cfg.log_positions)
        window_log = (
            log_gates[:, 1:],
            jnp.swapaxes(k_compact[:, :, 1:], 1, 2),
            jnp.stack(v_new_list, axis=0),
        )
        return out, new_recurrent_state, window_log
    return out, new_recurrent_state


def _heads_to_major(x: jax.Array, num_heads: int) -> jax.Array:
    """[1, rows, lanes] holding head h in lane h -> [num_heads, rows, 1]."""
    return fused_transpose_broadcast(x[..., :num_heads], src_dim=2, dst_dim=0)


def split_log_rows(
    rows: jax.Array,  # [rows_used, v_head_dim]: one slot's typed log rows
    count: jax.Array,  # scalar: logged updates that apply
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Splits one slot's draft log into the factored body's operands.

    Entries past `count` become identity updates (decay 1, key 0, update
    0), selected rather than multiplied so stale log rows cannot leak.

    Returns:
        decays [1, K, lanes] (lane h: value head h), keys [num_kq_heads, K,
        kq_head_dim] and updates [num_v_heads, K, v_head_dim].
    """
    num_pos, n_kq, n_v = cfg.log_positions, cfg.num_kq_heads, cfg.num_v_heads
    keys_row, updates_row, _ = config.log_layout_rows(num_pos, n_kq, n_v)
    d = rows.shape[-1]

    def live(shape):
        # Built at the operand shape: Mosaic does not broadcast boolean vectors.
        return jax.lax.broadcasted_iota(jnp.int32, shape, len(shape) - 2) < count

    decays = jnp.where(live((num_pos, d)), rows[:num_pos], 1.0)[jnp.newaxis]

    def per_head(first_row, num_heads, head):
        return jnp.concat(
            [
                rows[first_row + j * num_heads + head : first_row + j * num_heads + head + 1]
                for j in range(num_pos)
            ],
            axis=0,
        )

    keys = jnp.stack([per_head(keys_row, n_kq, h) for h in range(n_kq)], axis=0)
    updates = jnp.stack([per_head(updates_row, n_v, h) for h in range(n_v)], axis=0)
    keys = jnp.where(live((n_kq, num_pos, d)), keys, 0.0)
    updates = jnp.where(live((n_v, num_pos, d)), updates, 0.0)
    return decays, keys, updates


def factored_base_product(rows: jax.Array, base: jax.Array) -> jax.Array:
    """[num_v_heads, rows, kq_head_dim] . base -> [num_v_heads, rows, v_head_dim]."""
    return jax.lax.dot(
        rows,
        base,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def _num_columns(cfg: config.GDNConfig) -> int:
    """Coefficient columns: window updates, logged updates, the base, padding."""
    return -(-(cfg.chunk_size + cfg.log_positions + 1) // 8) * 8


def _base_column(cfg: config.GDNConfig) -> int:
    return cfg.chunk_size + cfg.log_positions


def _update_columns(cfg: config.GDNConfig) -> list[int]:
    """Columns with a rank-1 term in the state after window position 0:
    window position 0 and the logged updates."""
    return [0] + list(range(cfg.chunk_size, cfg.chunk_size + cfg.log_positions))


def factored_masks(cfg: config.GDNConfig) -> tuple[list[jax.Array], list[jax.Array], jax.Array]:
    """Float masks [1, 2W, N] shared by every sequence of a tile.

    Column j of a coefficient matrix is update `t_j` of the sequence
    (0-based, chronological): window position j for j < W, logged update
    j - W for W <= j < W + K, and the base (t = -1) in column W + K; later
    columns are padding. Row r < W is window key r, row r >= W window query
    r - W; both belong to update p_r = K + r mod W. The coefficient of
    column j in row r is the product of the decays m with t_j < m <= p_r.

    Returns:
        per decay m, where it enters a coefficient (1, else 0) and where it
        does not (1 - that), and the coefficients that exist (keys see the
        updates before their own, queries their own as well).
    """
    window, num_pos = cfg.chunk_size, cfg.log_positions
    num_rows, num_cols = 2 * window, _num_columns(cfg)
    shape = (1, num_rows, num_cols)
    r = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    j = jax.lax.broadcasted_iota(jnp.int32, shape, 2)
    p = num_pos + jnp.where(r < window, r, r - window)
    t = jnp.where(
        j < window,
        num_pos + j,
        jnp.where(j < window + num_pos, j - window, jnp.where(j == window + num_pos, -1, 1 << 20)),
    )
    enters = [jnp.where((t < m) & (m <= p), 1.0, 0.0) for m in range(num_pos + window)]
    stays = [1.0 - e for e in enters]
    valid = jnp.where(t < p + jnp.where(r < window, 0, 1), 1.0, 0.0)
    return enters, stays, valid


def factored_decay_products(
    decay_col: jax.Array,  # [num_v_heads, K + W, 1]: decays, chronological
    masks: tuple[list[jax.Array], list[jax.Array], jax.Array],
) -> jax.Array:
    """[num_v_heads, 2W, N]: every coefficient's product of decays.

    Each is a direct product (the empty product is 1), never a ratio of
    prefix products, which would be 0 / 0 once a decay or a prefix
    underflows. `enters * g + stays` is g or 1 exactly.
    """
    enters, stays, valid = masks
    coef = valid
    for m in range(decay_col.shape[1]):
        coef = coef * (enters[m] * decay_col[:, m : m + 1] + stays[m])
    return coef


def factored_key_dots(rows: jax.Array, keys: jax.Array) -> jax.Array:
    """[num_kq_heads, 2W, N]: the window's keys and queries against every key."""
    return jax.lax.dot(
        rows,
        keys,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def _update_terms(coef_row, keys, v_new, log_updates, cfg):
    """The base's decay and the (key, weighted update) rows of the rank-1
    terms of the state after window position 0."""
    cols = _update_columns(cfg)
    updates = [v_new[:, 0:1]] + [log_updates[:, j : j + 1] for j in range(cfg.log_positions)]
    key_rows = [keys[:, c : c + 1] for c in cols]
    weighted = [coef_row[:, :, c : c + 1] * u for c, u in zip(cols, updates)]
    base = _base_column(cfg)
    return coef_row[:, :, base : base + 1], key_rows, weighted


def _rows_to_8(parts: list[jax.Array]) -> jax.Array:
    """Concatenates [heads, n_i, dim] parts along rows, zero-padded to a multiple of 8."""
    rows = sum(x.shape[1] for x in parts)
    pad = -rows % 8
    if pad:
        parts = parts + [jnp.zeros((parts[0].shape[0], pad, parts[0].shape[2]), jnp.float32)]
    return jnp.concat(parts, axis=1)


def _bf16_split(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """x = hi + lo exactly, hi = x rounded to bf16 (nearest even), |lo| <= 2^-9 |x|.

    For finite x well inside the f32 range (the rounding increment cannot
    carry into the exponent's top).
    """
    bits = jax.lax.bitcast_convert_type(x, jnp.int32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & jnp.int32(-65536)
    hi = jax.lax.bitcast_convert_type(bits, jnp.float32)
    return hi, x - hi


def factored_update_pass(
    base: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    coef_row: jax.Array,  # [num_v_heads, 1, N]: coefficients of the state after position 0
    keys: jax.Array,  # [num_kq_heads, N, kq_head_dim]
    v_new: jax.Array,  # [num_v_heads, W, v_head_dim]
    log_updates: jax.Array,  # [num_v_heads, K, v_head_dim]
    cfg: config.GDNConfig,
) -> jax.Array:
    """The state after window position 0: base * decay + its rank-1 terms.

    The terms are one MXU product over [k_hi, k_hi, k_lo, k_lo] and [w_hi,
    w_lo, w_hi, w_lo], each operand split into its bf16 rounding and the
    remainder: the MXU rounds f32 operands to bf16 by default, so a plain
    product would carry a bf16 rounding of every term into the stored state.
    Split, only the remainders' own rounding is left, about 2^-17 of the
    terms. The 16 rows still fit one weight tile.
    """
    decay, key_rows, weighted = _update_terms(coef_row, keys, v_new, log_updates, cfg)
    k_hi, k_lo = _bf16_split(jnp.concat(key_rows, axis=1))
    w_hi, w_lo = _bf16_split(jnp.concat(weighted, axis=1))
    lhs = jnp.repeat(_rows_to_8([k_hi, k_hi, k_lo, k_lo]), cfg.v_per_kq_head, axis=0)
    rhs = _rows_to_8([w_hi, w_lo, w_hi, w_lo])
    return base * decay + jax.lax.dot(
        lhs,
        rhs,
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def factored_update_pass_bf16(base, coef_row, keys, v_new, log_updates, cfg):
    """`factored_update_pass` as a single MXU product (terms rounded to bf16)."""
    decay, key_rows, weighted = _update_terms(coef_row, keys, v_new, log_updates, cfg)
    lhs = jnp.repeat(_rows_to_8(key_rows), cfg.v_per_kq_head, axis=0)
    return base * decay + jax.lax.dot(
        lhs,
        _rows_to_8(weighted),
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def factored_update_pass_vpu(base, coef_row, keys, v_new, log_updates, cfg, block=32):
    """`factored_update_pass` on the VPU in f32, one block of state rows at a
    time so the block stays in registers across the terms."""
    decay, key_rows, weighted = _update_terms(coef_row, keys, v_new, log_updates, cfg)
    keys_t = jnp.swapaxes(_rows_to_8(key_rows), 1, 2)  # [num_kq_heads, kq_head_dim, 8]
    heads = []
    for h in range(cfg.num_v_heads):
        kq = h // cfg.v_per_kq_head
        blocks = []
        for r0 in range(0, cfg.kq_head_dim, block):
            blk = base[h, r0 : r0 + block] * decay[h]
            for i, w in enumerate(weighted):
                blk = blk + keys_t[kq, r0 : r0 + block, i : i + 1] * w[h]
            blocks.append(blk)
        heads.append(jnp.concat(blocks, axis=0))
    return jnp.stack(heads, axis=0)


def factored_window_vectors(
    proj: jax.Array,  # [num_v_heads, 2W, v_head_dim]: [k_c . B, q_c . B]
    rows: jax.Array,  # [num_kq_heads, 2W, kq_head_dim]: [k_c, q_c]
    keys: jax.Array,  # [num_kq_heads, N, kq_head_dim]: window keys, logged keys, 0
    v: jax.Array,  # [num_v_heads, W, v_head_dim]
    beta_row: jax.Array,  # [1, W, lanes]
    decay_row: jax.Array,  # [1, K + W, lanes]: logged then window decays
    log_updates: jax.Array,  # [num_v_heads, K, v_head_dim]
    masks: tuple[list[jax.Array], list[jax.Array], jax.Array],
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The window's updates and outputs from the base products, on vectors.

    See `factored_masks` for the coefficient layout.

    Returns:
        out [num_v_heads, W, v_head_dim], the window's update vectors v_new
        [num_v_heads, W, v_head_dim], and the coefficients [num_v_heads, 1,
        N] of the state after window position 0 (its decay in the base
        column, its rank-1 terms' in theirs).
    """
    window, num_pos, n_v = cfg.chunk_size, cfg.log_positions, cfg.num_v_heads
    base_col = _base_column(cfg)

    coef = factored_decay_products(_heads_to_major(decay_row, n_v), masks)
    decay = coef[:, :, base_col : base_col + 1]
    # Query row 0 holds the products from each update to window position 0.
    coef_row = coef[:, window : window + 1]
    coef = coef * jnp.repeat(factored_key_dots(rows, keys), n_v // cfg.num_kq_heads, axis=0)

    # Base and logged contributions: rows < W are k_c . (state before update
    # p_c, decayed by it), rows >= W the outputs before the window's own
    # updates. The contractions over updates run on the VPU: a matmul this
    # small still latches a full weight tile on the MXU.
    partial = decay * proj
    for j in range(num_pos):
        col = window + j
        partial = partial + coef[:, :, col : col + 1] * log_updates[:, j : j + 1]

    # Forward substitution of (I + beta L) v_new = beta (v - partial), L the
    # strictly lower coefficients among the window's own updates. After
    # iteration c, row c + 1 is final.
    beta = _heads_to_major(beta_row, n_v)  # [num_v_heads, W, 1]
    lower = beta * coef[:, :window, :window]
    v_new = beta * (v - partial[:, :window])
    for c in range(window - 1):
        v_new = v_new - lower[:, :, c : c + 1] * v_new[:, c : c + 1]

    out = partial[:, window:]
    for c in range(window):
        out = out + coef[:, window:, c : c + 1] * v_new[:, c : c + 1]
    return out, v_new, coef_row


def factored_products(
    q: jax.Array,  # [num_kq_heads, W, kq_head_dim]
    k: jax.Array,  # [num_kq_heads, W, kq_head_dim]
    log_keys: jax.Array,  # [num_kq_heads, K, kq_head_dim]
    base: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One sequence's window keys and queries, every update's key, and the
    product of the former with the base.

    Every state inside the window is the base times a decay plus rank-1
    terms of the updates applied since, so the window's products with
    those states follow from this one product with the base and products
    between vectors (`factored_window_vectors`).

    Returns:
        rows [num_kq_heads, 2W, kq_head_dim] ([k_c, q_c]), keys
        [num_kq_heads, N, kq_head_dim] (window keys, logged keys, 0) and
        proj [num_v_heads, 2W, v_head_dim] (rows . base).
    """
    pad = _num_columns(cfg) - cfg.chunk_size - cfg.log_positions
    rows = jnp.concat([k, q], axis=1)
    keys = jnp.concat(
        [k, log_keys, jnp.zeros((cfg.num_kq_heads, pad, cfg.kq_head_dim), jnp.float32)], axis=1
    )
    proj = factored_base_product(jnp.repeat(rows, cfg.v_per_kq_head, axis=0), base)
    return rows, keys, proj


def factored_gdn(
    real_sizes: jax.Array,  # [seq]
    q_large: jax.Array,  # [seq, num_kq_heads, W, kq_head_dim]
    k_large: jax.Array,  # [seq, num_kq_heads, W, kq_head_dim]
    v_large: jax.Array,  # [seq, num_v_heads, W, v_head_dim]
    b_large: jax.Array,  # [seq, 1, W, lanes]
    a_large: jax.Array,  # [seq, 1, W, lanes]
    base: jax.Array,  # [seq, num_v_heads, kq_head_dim, v_head_dim]
    log_rows: list[jax.Array],  # per seq [rows_used, v_head_dim]
    log_counts: list[jax.Array],  # per seq scalar: logged updates that apply
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
    order: str = "phased",
) -> tuple[jax.Array, jax.Array, list[jax.Array]]:
    """BATCHED LOG-mode window in factored form.

    The base is checkpoint block 0's state, the logged updates are the
    first `log_counts[s]` entries of the slot's log, and the result is what
    `recurrent_gdn` gives after folding them: the outputs, the state after
    window position 0, and the log of positions 1..K (as typed log rows).

    Per sequence: `factored_products`, `factored_window_vectors`, then
    `factored_update_pass`. `order` is the order the stages are traced in:
    "seq" one sequence after another, "phased" each stage for every sequence
    of the tile before the next stage, "pipelined" stage s of sequence i
    with stage s + 1 of sequence i - 1. The arithmetic is the same; phased
    is the fastest measured (0.54 us a request at tile 8, against 0.66 in
    sequence order and 0.73 pipelined).

    Returns:
        out [seq, W, num_v_heads, v_head_dim], state [seq, 1, num_v_heads,
        kq_head_dim, v_head_dim] and, per seq, the typed log rows
        [rows_used, v_head_dim] of window positions 1..K.
    """
    assert cfg.uses_factored_window and cfg.chunk_size == cfg.log_positions + 1
    assert b_large.shape[-1] == cfg.v_head_dim, "a gate row is one log row"
    compute = cfg.dtypes.compute
    window = cfg.chunk_size
    mask_dtype = get_mask_dtype(compute)
    iota = jax.lax.broadcasted_iota(mask_dtype, (cfg.seq_tile_size, 1, window, 1), 2)
    mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

    q_large = jnp.where(mask, q_large.astype(compute), 0)
    k_large = jnp.where(mask, k_large.astype(compute), 0)
    v_large = jnp.where(mask, v_large.astype(compute), 0)
    b_large = b_large.astype(compute)
    a_large = a_large.astype(compute)
    a_log = a_log.reshape(1, 1, 1, -1).astype(compute)
    dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(compute)

    q_large = l2_norm(q_large) * cfg.kq_head_dim**-0.5
    k_large = l2_norm(k_large)
    beta = jnp.where(mask, jax.nn.sigmoid(b_large), 0)
    # Masked rows decay by exp(0) = 1, so they leave the state as it is.
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_large + dt_bias)
    decays = jnp.exp(jnp.where(mask, gating_log, 0))

    masks = factored_masks(cfg)
    seqs = range(cfg.seq_tile_size)
    logs = [split_log_rows(log_rows[idx], log_counts[idx], cfg) for idx in seqs]

    def products(idx):
        return factored_products(q_large[idx], k_large[idx], logs[idx][1], base[idx], cfg)

    def vectors(idx, prod):
        rows, keys, proj = prod
        log_decays, _, log_updates = logs[idx]
        decay_row = jnp.concat([log_decays, decays[idx]], axis=1)
        return factored_window_vectors(
            proj, rows, keys, v_large[idx], beta[idx], decay_row, log_updates, masks, cfg
        )

    def update(idx, prod, vec):
        _, v_new, coef_row = vec
        return factored_update_pass(base[idx], coef_row, prod[1], v_new, logs[idx][2], cfg)

    num = cfg.seq_tile_size
    if order == "phased":
        prods = [products(idx) for idx in seqs]
        vecs = [vectors(idx, prods[idx]) for idx in seqs]
        states = [update(idx, prods[idx], vecs[idx]) for idx in seqs]
    elif order == "pipelined":
        prods, vecs, states = [None] * num, [None] * num, [None] * num
        for t in range(num + 2):
            if t < num:
                prods[t] = products(t)
            if 0 <= t - 1 < num:
                vecs[t - 1] = vectors(t - 1, prods[t - 1])
            if 0 <= t - 2 < num:
                states[t - 2] = update(t - 2, prods[t - 2], vecs[t - 2])
    else:
        assert order == "seq", order
        prods, vecs, states = [], [], []
        for idx in seqs:
            prods.append(products(idx))
            vecs.append(vectors(idx, prods[idx]))
            states.append(update(idx, prods[idx], vecs[idx]))

    out_list = [vec[0].swapaxes(0, 1) for vec in vecs]
    state_list = states
    log_list = []
    for idx in seqs:
        k_idx, v_new = k_large[idx], vecs[idx][1]
        log_list.append(
            jnp.concat(
                [decays[idx, 0, 1:]]
                + [k_idx[h, j : j + 1] for j in range(1, window) for h in range(cfg.num_kq_heads)]
                + [v_new[h, j : j + 1] for j in range(1, window) for h in range(cfg.num_v_heads)],
                axis=0,
            )
        )
    out = jnp.stack(out_list, axis=0)
    state = jnp.stack(state_list, axis=0)[:, jnp.newaxis]
    return out, state, log_list
