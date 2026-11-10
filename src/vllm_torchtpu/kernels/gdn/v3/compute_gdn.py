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
import os

import jax
import jax.numpy as jnp

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


def _chunk_dot(lhs: jax.Array, rhs: jax.Array, dimension_numbers) -> jax.Array:
    """MXU dot of the chunked path with f32 accumulation; the operand
    precision follows TPU_GDN_PREFILL_DOT_PRECISION (config.prefill_dot_precision)."""
    mode = config.prefill_dot_precision()
    precision = None
    if mode == "bf16":
        lhs = lhs.astype(jnp.bfloat16)
        rhs = rhs.astype(jnp.bfloat16)
    elif mode == "fp32":
        precision = jax.lax.Precision.HIGHEST
    return jax.lax.dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def _stub_dot(lhs: jax.Array, rhs: jax.Array, out_shape: tuple[int, ...]) -> jax.Array:
    """VPU stand-in of a batched dot for the TPU_GDN_BENCH_STUB=dots
    attribution: an f32 array of the dot's shape that depends on both
    operands, built from lane slices and leading-dim repeats, no MXU work.
    Wrong numerics by design."""
    h, m, n = out_shape

    def fit(x, rows, cols):
        # Rows: slice or tile along the second-minor dim; cols: lane slice or
        # lane broadcast of column 0.
        if x.shape[1] >= rows:
            x = x[:, :rows]
        else:
            reps = -(-rows // x.shape[1])
            x = jnp.concatenate([x] * reps, axis=1)[:, :rows]
        if x.shape[2] >= cols:
            x = x[:, :, :cols]
        else:
            x = jnp.broadcast_to(x[:, :, :1], (x.shape[0], rows, cols))
        return x.astype(jnp.float32)

    return fit(lhs, m, n) + fit(rhs, m, n)


def invert_unit_lower_doubling(t: jax.Array, identity: jax.Array) -> jax.Array:
    """Inverse of the unit lower triangular [heads, chunk, chunk] `t` as the
    product (I - L)(I + L^2)(I + L^4)... with L = t - I, which is exact once
    L^(2^k) vanishes (L is nilpotent of index chunk). Each factor costs one
    squaring and one product on the MXU; the final (I + L^(2^k)) with
    2^k >= chunk is the identity and is skipped."""
    chunk = t.shape[-1]
    lower = t - identity
    acc = identity - lower
    power = lower
    dims = (((2,), (1,)), ((0,), (0,)))
    k = 2
    while k < chunk:
        power = _chunk_dot(power, power, dims).astype(t.dtype)
        acc = acc + _chunk_dot(acc, power, dims).astype(t.dtype)
        k *= 2
    return acc


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
            prev_sum = _chunk_dot(
                interaction_t,
                solved_x,
                (((2,), (1,)), ((0,), (0,))),
            )
            target_b = e_block - prev_sum

        # NOTE: Utilize fp32 to minimize cost of sublane rolling.
        local_t = t[:, start:end, start:end].astype(jnp.float32)
        x_block = local_forward_sub(local_t, target_b)
        x_blocks.append(x_block.astype(out_dtype))

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

    gates = dict(
        beta_large=beta_large,
        gating_map=gating_map,
        gating_map_masked=gating_map_masked,
        gating_backward=gating_backward,
        gating_forward=gating_forward,
        gating_last=gating_last,
        identity_mask=identity_mask,
        lower_mask=lower_mask,
    )

    group = config.prefill_head_group()
    vpk = cfg.v_per_kq_head
    if group <= 0 or group >= cfg.num_v_heads or group % vpk:
        return _chunked_gdn_heads(q_large, k_large, v_large, state_prev, gates, cfg)

    # Head groups run one after another so the live intermediates of the
    # chunk are those of one group.
    outs, states = [], []
    for h0 in range(0, cfg.num_v_heads, group):
        h1 = h0 + group
        sub = {k: v[h0:h1] for k, v in gates.items()}
        out, state = _chunked_gdn_heads(
            q_large[h0 // vpk : h1 // vpk],
            k_large[h0 // vpk : h1 // vpk],
            v_large[h0:h1],
            state_prev[h0:h1],
            sub,
            cfg,
        )
        outs.append(out)
        states.append(state)
    return jnp.concatenate(outs, axis=0), jnp.concatenate(states, axis=0)


def _chunked_gdn_heads(
    q_large: jax.Array,  # [kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [v_heads, chunk, v_head_dim]
    state_prev: jax.Array,  # [v_heads, kq_head_dim, v_head_dim]
    gates: dict,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """The chunk's dots for a slice of value heads (and their kq heads)."""
    stub = config.bench_stub()
    chunk = cfg.chunk_size
    dk, dv = cfg.kq_head_dim, cfg.v_head_dim
    n_v = v_large.shape[0]
    beta_large = gates["beta_large"]
    gating_map = gates["gating_map"]
    gating_backward = gates["gating_backward"]
    gating_forward = gates["gating_forward"]
    gating_last = gates["gating_last"]

    # NOTE: Repeat along non lane/sublane dim is free.
    q_repeat = jnp.repeat(q_large, cfg.v_per_kq_head, axis=0)
    k_repeat = jnp.repeat(k_large, cfg.v_per_kq_head, axis=0)

    # [num_v_heads, chunk, kq_head_dim]
    k_beta_repeat = k_repeat * beta_large

    # [num_v_heads, chunk, chunk]
    if stub == "dots":
        beta_k_k_t = _stub_dot(k_beta_repeat, k_repeat, (n_v, chunk, chunk))
    elif config.prefill_shared_kkt():
        # (beta k) k^T = diag(beta) (k k^T): one dot per kq head, the rows
        # scaled per value head.
        kk_t = _chunk_dot(k_large, k_large, (((2,), (2,)), ((0,), (0,))))
        beta_k_k_t = jnp.repeat(kk_t.astype(cfg.dtypes.compute), cfg.v_per_kq_head, axis=0)
        beta_k_k_t = beta_k_k_t * beta_large
    else:
        beta_k_k_t = _chunk_dot(k_beta_repeat, k_repeat, (((2,), (2,)), ((0,), (0,))))
    beta_k_k_t = beta_k_k_t.astype(cfg.dtypes.compute)
    gating_beta_k_k_t = gates["gating_map_masked"] * beta_k_k_t
    t = jnp.where(gates["identity_mask"], 1, gating_beta_k_k_t)

    # [num_v_heads, chunk, chunk]
    if stub == "tinv":
        t_inv = t
    elif config.prefill_tinv_mode() == "doubling":
        identity = jnp.where(gates["identity_mask"], 1.0, 0.0).astype(cfg.dtypes.compute)
        t_inv = invert_unit_lower_doubling(t, identity)
    else:
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
    if stub == "dots":
        merged_uw = _stub_dot(t_inv, merged_v_k, (n_v, chunk, dv + dk))
    else:
        merged_uw = _chunk_dot(t_inv, merged_v_k, (((2,), (1,)), ((0,), (0,))))
    merged_uw = merged_uw.astype(cfg.dtypes.compute)

    # [num_v_heads, chunk, v_head_dim]
    u, w = jnp.split(merged_uw, [dv], axis=-1)

    # [num_v_heads, chunk, kq_head_dim]
    q_large_gating = q_repeat * gating_forward
    # NOTE: Concatenate lhs with same rhs to leverage weight
    # stationary architecture.
    # [num_v_heads, 2 * chunk, kq_head_dim]
    merged_w_q = jnp.concat([w, q_large_gating], axis=1)
    # [num_v_heads, 2 * chunk, v_head_dim]
    if stub == "dots":
        merged_ws_out_updated = _stub_dot(merged_w_q, state_prev, (n_v, 2 * chunk, dv))
    else:
        merged_ws_out_updated = _chunk_dot(
            merged_w_q, state_prev, (((2,), (1,)), ((0,), (0,)))
        )

    # NOTE: Splitting along non sublane/lane dim is free.
    ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
    ws = ws.astype(cfg.dtypes.compute)

    # [num_v_heads, chunk, v_head_dim]
    u_ws = u - ws

    # [num_v_heads, chunk, kq_head_dim]
    k_repeat_gating = k_repeat * gating_backward

    # [num_v_heads, kq_head_dim, v_head_dim]
    if stub == "dots":
        state_new = _stub_dot(k_repeat_gating, u_ws, (n_v, dk, dv))
    else:
        state_new = _chunk_dot(k_repeat_gating, u_ws, (((1,), (1,)), ((0,), (0,))))

    # [num_v_heads, kq_head_dim, v_head_dim]
    state_updated = state_prev * gating_last
    state = state_updated + state_new

    # [num_kq_heads, chunk, chunk]
    if stub == "dots":
        out_qk = _stub_dot(q_large, k_large, (q_large.shape[0], chunk, chunk))
    else:
        out_qk = _chunk_dot(q_large, k_large, (((2,), (2,)), ((0,), (0,))))
    out_qk = out_qk.astype(cfg.dtypes.compute)
    # NOTE: must perform repeat after matmul to reduce required compute.
    # [num_v_heads, chunk, chunk]
    out_qk = jnp.repeat(out_qk, cfg.v_per_kq_head, axis=0)
    out_qk *= gating_map
    out_qk = jnp.where(gates["lower_mask"], out_qk, 0)

    # [num_v_heads, chunk, v_head_dim]
    if stub == "dots":
        out_new = _stub_dot(out_qk, u_ws, (n_v, chunk, dv))
    else:
        out_new = _chunk_dot(out_qk, u_ws, (((2,), (1,)), ((0,), (0,))))
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
        if config.bench_stub() == "chunked":
            # Attribution stand-in: no chunk compute, the state passes through.
            out = v_large[idx]
            state = state_prev[idx]
        else:
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
    k_compact_t: jax.Array | None,  # [num_kq_heads, chunk, kq_head_dim, 1]
    v_compact: jax.Array,  # [num_v_heads, chunk, 1, v_head_dim]
    gating_log: jax.Array,  # [num_v_heads, chunk, 1, 1]
    beta: jax.Array,  # [num_v_heads, chunk, 1, 1]
    state: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfgs: config.GDNConfig,
    emit_state=None,
) -> tuple[jax.Array, jax.Array | None]:
    """Perform recurrent GDN over input [num_heads, chunk, 1, head_dim].

    The returned state stacks the post-token state of the last
    `cfgs.window_size` window positions ([window_size, num_v_heads,
    kq_head_dim, v_head_dim]); with `window_size == 1` that is just the final
    state. With `emit_state`, each of those states is handed to
    `emit_state(t, state)` as the loop produces it and None is returned in
    their place.
    """
    use_fused_1dot = os.environ.get("TPU_GDN_FUSED_1DOT", "0") == "1"
    stub = config.bench_stub()
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

        if k_compact_t is None:
            # kb[h, r, :] = k[h, r]: k^T already broadcast along lanes, one
            # sublane broadcast and XLU transpose per kq head, shared by its
            # value heads through the free leading-dim repeat.
            k_rows = k_compact[:, c_idx]  # [num_kq_heads, 1, kq_head_dim]
            kb = jnp.broadcast_to(
                k_rows, (cfgs.num_kq_heads, cfgs.v_head_dim, cfgs.kq_head_dim)
            )
            k_curr_t = jnp.repeat(
                jnp.transpose(kb, (0, 2, 1)), cfgs.v_per_kq_head, axis=0
            )
        else:
            # [num_v_heads, kq_head_dim, 1]
            k_curr_t = k_compact_t[:, c_idx]
            k_curr_t = jnp.repeat(k_curr_t, cfgs.v_per_kq_head, axis=0)

        # [num_v_heads, 1, 1]
        beta_curr = beta[:, c_idx]
        gating_curr = gating_log[:, c_idx]

        if stub == "recurrence":
            out = v_curr.astype(cfgs.dtypes.compute)
        elif use_fused_1dot:
            iota_in = jax.lax.broadcasted_iota(
                jnp.int32, (cfgs.num_v_heads, 8, cfgs.kq_head_dim), 1
            )
            kq_curr = jnp.where(
                iota_in == 0, k_curr, jnp.where(iota_in == 1, q_curr, 0.0)
            )
            if stub == "dots":
                kq_dot = (kq_curr * state[:, :8, :]).astype(cfgs.dtypes.compute)
            else:
                kq_dot = jax.lax.dot(
                    kq_curr,
                    state,
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                ).astype(cfgs.dtypes.compute)
            if config.dec_rowpick():
                # DGDN L6: the masked sum below adds seven +0.0 to one row,
                # which is that row + 0.0 exactly (-0.0 -> +0.0, else same).
                k_state = kq_dot[:, 0:1, :] + 0.0
                q_state = kq_dot[:, 1:2, :] + 0.0
            else:
                iota_out = jax.lax.broadcasted_iota(jnp.int32, kq_dot.shape, 1)
                k_state = jnp.where(iota_out == 0, kq_dot, 0.0).sum(
                    axis=1, keepdims=True
                )
                q_state = jnp.where(iota_out == 1, kq_dot, 0.0).sum(
                    axis=1, keepdims=True
                )
            v_updated = k_state * gating_curr
            v_diff = v_curr - v_updated
            v_new = beta_curr * v_diff
            state_new = k_curr_t * v_new
            state = state * gating_curr + state_new
            qk_dot = jnp.sum(q_curr * k_curr, axis=2, keepdims=True)
            out = (q_state * gating_curr + qk_dot * v_new).astype(cfgs.dtypes.compute)
        else:
            # [num_v_heads, kq_head_dim, v_head_dim]
            state_updated = state * gating_curr

            # [num_v_heads, 1, v_head_dim]
            if stub == "dots":
                v_updated = (k_curr * state_updated[:, :1, :]).astype(
                    cfgs.dtypes.compute
                )
            else:
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
            if stub == "dots":
                out = (q_curr * state[:, :1, :]).astype(cfgs.dtypes.compute)
            else:
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
            if emit_state is None:
                state_list.append(state)
            else:
                emit_state(c_idx - (cfgs.chunk_size - cfgs.window_size), state)

    out = jnp.stack(out_list, axis=0)
    if emit_state is not None:
        return out, None
    return out, jnp.stack(state_list, axis=0)


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
    emit_state=None,
) -> tuple[jax.Array, jax.Array | None]:
    """Perform recurrent GDN over input [seq, num_heads, chunk, 1, head_dim].

    The returned state has one checkpoint per window position: [seq,
    window_size, num_v_heads, kq_head_dim, v_head_dim]. Positions >=
    real_sizes repeat the last valid state; they are never written back to
    HBM. With `emit_state`, checkpoint t of sequence idx is handed to
    `emit_state(idx, t, state)` as it is produced and None is returned in
    place of the stacked states; `state_prev` may then be a sequence of
    per-slot states rather than one stacked array.
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
    if config.kt_xpose():
        # The per-token transposed broadcast is built inside the chain.
        k_compact_t = None
    else:
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

    out_list = []
    new_state_list = []

    for idx in range(cfg.seq_tile_size):
        out, state = recurrent_gdn_per_seq(
            q_compact[idx],
            k_compact[idx],
            None if k_compact_t is None else k_compact_t[idx],
            v_compact[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfg,
            emit_state=(
                None
                if emit_state is None
                else functools.partial(_emit_for_slot, emit_state, idx)
            ),
        )
        out_list.append(out)
        new_state_list.append(state)

    out = jnp.stack(out_list, axis=0)
    if emit_state is not None:
        return out, None
    new_recurrent_state = jnp.stack(new_state_list, axis=0)

    return out, new_recurrent_state


def _emit_for_slot(emit_state, idx: int, t: int, state: jax.Array) -> None:
    emit_state(idx, t, state)


def recurrent_gdn_grouped(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
    group: int,
    load_state,
    emit_state,
    after_seq=None,
) -> jax.Array:
    """Recurrent GDN with the token chain run one head group at a time.

    Head groups are the outer loop and the window tokens the inner one, so a
    group's state is decoded once (`load_state(idx, h0, nh)` returns the
    f32 [nh, kq_head_dim, v_head_dim] state of value heads [h0, h0 + nh) of
    slot idx), stays live across the window, and every checkpoint is handed
    to `emit_state(idx, t, h0, nh, state)` as soon as that group has
    produced it. The kq rows a group shares (its heads belong to one kq head,
    or the group spans whole kq heads) are read, masked and transposed once
    per token for the whole group. Returns the output [seq, chunk,
    num_v_heads, v_head_dim]; the states only ever reach emit_state.
    """
    assert os.environ.get("TPU_GDN_FUSED_1DOT", "0") == "1", (
        "the grouped recurrence runs the fused 1-dot form only"
    )
    vpk = cfg.v_per_kq_head
    assert cfg.num_v_heads % group == 0, (cfg.num_v_heads, group)
    assert vpk % group == 0 or group % vpk == 0, (vpk, group)
    stub = config.bench_stub()
    xpose = config.kt_xpose()
    dk, dv = cfg.kq_head_dim, cfg.v_head_dim

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

    q_compact = l2_norm(q_compact)
    q_compact *= cfg.kq_head_dim**-0.5
    k_compact = l2_norm(k_compact)

    beta = jax.nn.sigmoid(b_compact)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_compact + dt_bias)
    beta = jnp.where(mask, beta, 0)
    # A masked row decays by exp(0) = 1 and adds nothing, so the state holds.
    gating_log = jnp.where(mask, gating_log, 0)
    gating = jnp.exp(gating_log)
    # [seqs, num_v_heads, chunk, 1, 1]
    beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)[:, : cfg.num_v_heads]
    gating = fused_transpose_broadcast(gating, src_dim=4, dst_dim=1)
    gating = gating[:, : cfg.num_v_heads]

    first_emit = cfg.chunk_size - cfg.window_size
    iota_in = jax.lax.broadcasted_iota(jnp.int32, (8, dk), 0)
    out_rows = []
    for idx in range(cfg.seq_tile_size):
        outs = [[] for _ in range(cfg.chunk_size)]
        for h0 in range(0, cfg.num_v_heads, group):
            state = load_state(idx, h0, group)
            j0, j1 = h0 // vpk, (h0 + group - 1) // vpk + 1
            for c in range(cfg.chunk_size):
                new_blocks = []
                for j in range(j0, j1):
                    hs = slice(max(h0, j * vpk) - h0, min(h0 + group, (j + 1) * vpk) - h0)
                    nj = hs.stop - hs.start
                    s_blk = state[hs]
                    # [num_v_heads, 1, v_head_dim] pieces of this kq head.
                    v_curr = v_compact[idx, h0 + hs.start : h0 + hs.stop, c]
                    beta_c = beta[idx, h0 + hs.start : h0 + hs.stop, c]
                    g_c = gating[idx, h0 + hs.start : h0 + hs.stop, c]
                    if stub == "recurrence":
                        outs[c].append(v_curr[:, 0, :])
                        new_blocks.append(s_blk)
                        continue
                    q_row = q_compact[idx, j, c]  # [1, kq_head_dim]
                    k_row = k_compact[idx, j, c]
                    # Rows 0 and 1 of one 8-row MXU operand carry k and q.
                    lhs = jnp.where(iota_in == 0, k_row, jnp.where(iota_in == 1, q_row, 0.0))
                    lhs = jnp.broadcast_to(lhs[None], (nj, 8, dk))
                    if xpose:
                        # kb[r, :] = k[r]: the lane broadcast of k^T, built once
                        # per kq head and token and shared by its value heads.
                        kb = jnp.transpose(jnp.broadcast_to(k_row, (dv, dk)))[None]
                    else:
                        # [1, kq_head_dim, 1]; the lane broadcast happens in
                        # the outer product below, once per value head.
                        kb = fused_transpose_broadcast(
                            k_row.reshape(1, 1, dk), src_dim=2, dst_dim=1
                        )
                    qk = jnp.sum(q_row * k_row, axis=-1, keepdims=True)  # [1, 1]
                    if stub == "dots":
                        kq_dot = (lhs * s_blk[:, :8, :]).astype(cfg.dtypes.compute)
                    else:
                        kq_dot = jax.lax.dot(
                            lhs,
                            s_blk,
                            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                            preferred_element_type=jnp.float32,
                        ).astype(cfg.dtypes.compute)
                    iota_out = jax.lax.broadcasted_iota(jnp.int32, kq_dot.shape, 1)
                    k_state = jnp.where(iota_out == 0, kq_dot, 0.0).sum(
                        axis=1, keepdims=True
                    )
                    q_state = jnp.where(iota_out == 1, kq_dot, 0.0).sum(
                        axis=1, keepdims=True
                    )
                    v_new = beta_c * (v_curr - k_state * g_c)
                    s_blk = s_blk * g_c + kb * v_new
                    out = (q_state * g_c + qk * v_new).astype(cfg.dtypes.compute)
                    outs[c].append(out[:, 0, :])
                    new_blocks.append(s_blk)
                state = new_blocks[0] if len(new_blocks) == 1 else jnp.concat(new_blocks, axis=0)
                if c >= first_emit:
                    emit_state(idx, c - first_emit, h0, group, state)
        if after_seq is not None:
            after_seq(idx)
        out_rows.append(jnp.stack([jnp.concat(o, axis=0) for o in outs], axis=0))
    return jnp.stack(out_rows, axis=0)
