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

from collections.abc import Callable, Sequence
from typing import NamedTuple, overload

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import block_doubling, config, triangular_solve


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
    if cfg.use_block_doubling:
        out, state = block_doubling.run_chunk_per_seq(
            q_large, k_large, v_large, gating_log, beta, state_prev, cfg
        )
        assert out is not None and state is not None
        return out, state

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

    # Contract unscaled [K; Q] against state_prev before the triangular solve so
    # the solve operates on the single v_head_dim-wide residual v_diff.
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
        triangular_solve.solve_triangular_rhs(
            gating_beta_k_k_t, v_diff, block_size=cfg.triangular_block_size
        )
        * beta_large
    )

    # Absorb gating_backward into u_ws so sibling V heads share the same
    # unscaled K, letting the compiler reuse K and its transpose per KQ head.
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


class _RecurrentStep(NamedTuple):
    """Precomputed per-token inputs for one recurrent GDN step."""

    q: jax.Array
    k: jax.Array
    v: jax.Array
    beta: jax.Array
    gating: jax.Array


def _make_in_memory_state_io(
    state_prev: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[
    Callable[[int, int], jax.Array],
    Callable[[int, int, int, jax.Array], None],
    Callable[[], jax.Array],
]:
    """Adapts an in-memory `state_prev` tensor to per-head streaming callbacks."""
    ckpts: list[list[list[jax.Array]]] = [
        [[] for _ in range(cfg.window_size)] for _ in range(cfg.seq_tile_size)
    ]

    def load_head(s: int, v: int) -> jax.Array:
        return state_prev[s, v].astype(cfg.dtypes.compute)

    def store_ckpt(s: int, v: int, w_idx: int, state_j: jax.Array) -> None:
        assert v == len(ckpts[s][w_idx]), (
            f"store_ckpt requires ascending head order; expected head "
            f"{len(ckpts[s][w_idx])}, got {v}."
        )
        ckpts[s][w_idx].append(state_j)

    def collect_states() -> jax.Array:
        return jnp.stack(
            [
                jnp.stack(
                    [jnp.stack(ckpts[s][w], axis=0) for w in range(cfg.window_size)],
                    axis=0,
                )
                for s in range(cfg.seq_tile_size)
            ],
            axis=0,
        )

    return load_head, store_ckpt, collect_states


def _prepare_recurrent_steps(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> list[_RecurrentStep]:
    """Precomputes masked, normalized per-token inputs for recurrent GDN."""
    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2
    )
    mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

    a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    q_scale = cfg.kq_head_dim**-0.5

    steps: list[_RecurrentStep] = []
    for c_idx in range(cfg.chunk_size):
        mask_c = mask[:, :, c_idx]
        q_c = jnp.where(mask_c, q_compact[:, :, c_idx].astype(cfg.dtypes.compute), 0)
        k_c = jnp.where(mask_c, k_compact[:, :, c_idx].astype(cfg.dtypes.compute), 0)
        v_c = jnp.where(mask_c, v_compact[:, :, c_idx].astype(cfg.dtypes.compute), 0)
        q_c = l2_norm(q_c) * q_scale
        k_c = l2_norm(k_c)

        b_c = b_compact[:, :, c_idx].astype(cfg.dtypes.compute)
        a_c = a_compact[:, :, c_idx].astype(cfg.dtypes.compute)
        beta_c = jnp.where(mask_c, jax.nn.sigmoid(b_c), 0)
        # Masked gating_c evaluates to exp(0) = 1 (no decay on invalid tokens).
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
        steps.append(
            _RecurrentStep(
                q=q_c,
                k=k_c,
                v=v_c,
                beta=beta_curr,
                gating=gating_curr,
            )
        )
    return steps


def _run_single_token_recurrent(
    step: _RecurrentStep,
    load_head: Callable[[int, int], jax.Array],
    store_ckpt: Callable[[int, int, int, jax.Array], None],
    cfg: config.GDNConfig,
    sub_block_heads: int,
) -> jax.Array:
    """Streams single-token decode `sub_block_heads` value heads at a time.

    Bounding live head state to `sub_block_heads` (default 2) keeps vector
    register usage O(sub_block_heads) instead of O(num_v_heads), preventing
    register spills when packed state formats unpack heads on the fly.
    """
    dot_dims = (((2,), (1,)), ((0,), (0,)))
    # Concatenate [k, q] along axis=2 so each head pair issues one MXU dot
    # instead of two dependent contractions against the recurrent state.
    kq = jnp.concat([step.k, step.q], axis=2)
    qk_dot = jnp.sum(
        step.q.astype(jnp.float32) * step.k.astype(jnp.float32),
        axis=-1,
        keepdims=True,
    )
    seq_outs: list[jax.Array] = []
    for s in range(cfg.seq_tile_size):
        k_t_per_kq: list[jax.Array] | None = None
        head_outs: list[jax.Array] = []
        for sub0 in range(0, cfg.num_v_heads, sub_block_heads):
            sub_blk = min(sub_block_heads, cfg.num_v_heads - sub0)
            states_sub = [load_head(s, sub0 + j) for j in range(sub_blk)]
            bf16_state = states_sub[0].dtype == jnp.bfloat16
            if k_t_per_kq is None:
                k_t_per_kq = [
                    fused_transpose_broadcast(
                        step.k[s : s + 1, kq_idx : kq_idx + 1], src_dim=3, dst_dim=2
                    )[0, 0]
                    for kq_idx in range(cfg.num_kq_heads)
                ]
                if bf16_state:
                    k_t_per_kq = [k_t.astype(jnp.bfloat16) for k_t in k_t_per_kq]
            if bf16_state:
                # Defer scalar decay past the MXU dot so the projection keeps
                # float32 decay precision; state_upd rounds decay to bfloat16
                # (decays above 1 - 2**-9 become 1).
                mxu_rhs = jnp.stack(states_sub, axis=0)
                state_upd_list = [
                    st * step.gating[s, sub0 + j].astype(jnp.bfloat16)
                    for j, st in enumerate(states_sub)
                ]
            else:
                state_upd_list = [
                    st * step.gating[s, sub0 + j] for j, st in enumerate(states_sub)
                ]
                mxu_rhs = jnp.stack(state_upd_list, axis=0)
            kq_sub = jnp.stack(
                [kq[s, (sub0 + j) // cfg.v_per_kq_head] for j in range(sub_blk)],
                axis=0,
            )
            if bf16_state:
                kq_sub = kq_sub.astype(jnp.bfloat16)
            kq_proj = jax.lax.dot_general(
                kq_sub,
                mxu_rhs,
                dimension_numbers=dot_dims,
                preferred_element_type=jnp.float32,
            )

            for j in range(sub_blk):
                v = sub0 + j
                kq_idx = v // cfg.v_per_kq_head
                proj_j = (
                    kq_proj[j] * step.gating[s, v] if bf16_state else kq_proj[j]
                ).astype(cfg.dtypes.compute)
                v_updated_j = proj_j[:1, :]
                out_decayed_j = proj_j[1:, :]
                v_new_j = step.beta[s, v] * (step.v[s, v] - v_updated_j)
                out_j = (
                    out_decayed_j.astype(jnp.float32)
                    + qk_dot[s, kq_idx] * v_new_j.astype(jnp.float32)
                ).astype(cfg.dtypes.compute)
                head_outs.append(out_j)
                v_for_upd = v_new_j.astype(jnp.bfloat16) if bf16_state else v_new_j
                state_j = state_upd_list[j] + k_t_per_kq[kq_idx] * v_for_upd
                store_ckpt(s, v, 0, state_j)

        seq_outs.append(jnp.concatenate(head_outs, axis=0)[jnp.newaxis])
    return jnp.stack(seq_outs, axis=0)


def _run_multi_token_recurrent(
    steps: Sequence[_RecurrentStep],
    state_prev: jax.Array | None,
    load_head: Callable[[int, int], jax.Array],
    store_ckpt: Callable[[int, int, int, jax.Array], None],
    cfg: config.GDNConfig,
) -> jax.Array:
    """Runs multi-token recurrent steps per sequence across all value heads.

    Uses the two-dot `recurrent_gdn_per_seq` update (`k @ S_updated`, rank-1
    state update, then `q @ S`) one sequence at a time so peak live state stays
    O(num_v_heads) instead of O(seq_tile_size * num_v_heads) while keeping
    verify outputs numerically identical to `recurrent_gdn_per_seq`.

    Args:
        steps: Precomputed per-token recurrent inputs.
        state_prev: Optional prior state `[seq, num_v_heads, kq_dim, v_dim]`.
        load_head: Per-head loader `(s, v) -> [kq_dim, v_dim]` when
            `state_prev` is `None`.
        store_ckpt: Per-head checkpoint writer `(s, v, w_idx, state)` called in
            ascending head order per `(s, w_idx)`.
        cfg: GDN kernel configuration.

    Returns:
        Stacked outputs `[seq, chunk_size, num_v_heads, v_head_dim]`.
    """
    seq_dot_dims = (((2,), (1,)), ((0,), (0,)))
    ckpt_base = cfg.chunk_size - cfg.window_size
    out_per_seq: list[jax.Array] = []
    for s in range(cfg.seq_tile_size):
        state_s = (
            state_prev[s]
            if state_prev is not None
            else jnp.stack([load_head(s, v) for v in range(cfg.num_v_heads)], axis=0)
        )
        seq_out_list: list[jax.Array] = []
        for c_idx, step in enumerate(steps):
            state_updated = state_s * step.gating[s]
            k_curr = jnp.repeat(step.k[s], cfg.v_per_kq_head, axis=0)
            v_updated = jax.lax.dot_general(
                k_curr,
                state_updated,
                dimension_numbers=seq_dot_dims,
                preferred_element_type=jnp.float32,
            ).astype(cfg.dtypes.compute)
            v_new = step.beta[s] * (step.v[s] - v_updated)
            k_curr_t = jnp.repeat(
                fused_transpose_broadcast(step.k[s : s + 1], src_dim=3, dst_dim=2)[0],
                cfg.v_per_kq_head,
                axis=0,
            )
            state_s = state_updated + k_curr_t * v_new
            q_curr = jnp.repeat(step.q[s], cfg.v_per_kq_head, axis=0)
            out_s = jax.lax.dot_general(
                q_curr,
                state_s,
                dimension_numbers=seq_dot_dims,
                preferred_element_type=jnp.float32,
            ).astype(cfg.dtypes.compute)
            seq_out_list.append(out_s[:, 0, :])
            if c_idx >= ckpt_base:
                w_idx = c_idx - ckpt_base
                for v in range(cfg.num_v_heads):
                    store_ckpt(s, v, w_idx, state_s[v])
        out_per_seq.append(jnp.stack(seq_out_list, axis=0))
    return jnp.stack(out_per_seq, axis=0)


@overload
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
    *,
    load_head_fn: None = None,
    store_ckpt_head_fn: None = None,
    sub_block_heads: int = 2,
) -> tuple[jax.Array, jax.Array]: ...


@overload
def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
    *,
    load_head_fn: Callable[[int, int], jax.Array],
    store_ckpt_head_fn: Callable[[int, int, int, jax.Array], None],
    sub_block_heads: int = 2,
) -> tuple[jax.Array, None]: ...


def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
    *,
    load_head_fn: Callable[[int, int], jax.Array] | None = None,
    store_ckpt_head_fn: Callable[[int, int, int, jax.Array], None] | None = None,
    sub_block_heads: int = 2,
) -> tuple[jax.Array, jax.Array | None]:
    """Performs recurrent GDN, streaming single-token decode in value-head sub-blocks.

    Single-token decode (`chunk_size == 1`) streams `sub_block_heads` value
    heads at a time in `_run_single_token_recurrent`. Multi-token verify
    windows (`chunk_size > 1`) run token-outer across all value heads in
    `_run_multi_token_recurrent`.

    Args:
        real_sizes: Valid token counts per sequence `[seq]`.
        q_compact: Query tensor `[seq, num_kq_heads, chunk, 1, kq_head_dim]`.
        k_compact: Key tensor `[seq, num_kq_heads, chunk, 1, kq_head_dim]`.
        v_compact: Value tensor `[seq, num_v_heads, chunk, 1, v_head_dim]`.
        b_compact: Beta-gate pre-activations
            `[seq, 1, chunk, 1, padded_v_heads]`.
        a_compact: Decay-gate pre-activations
            `[seq, 1, chunk, 1, padded_v_heads]`.
        state_prev: In-memory prior state
            `[seq, num_v_heads, kq_dim, v_dim]`, or `None` to stream per-head
            state via `load_head_fn` and `store_ckpt_head_fn`.
        a_log: Per-head log-decay parameter.
        dt_bias: Per-head decay bias parameter.
        cfg: GDN kernel configuration.
        load_head_fn: Per-head loader `(s, v) -> [kq_dim, v_dim]` when
            `state_prev` is `None`. On single-token decode (`chunk_size == 1`),
            returning `bfloat16` keeps the MXU projection and state update in
            `bfloat16`.
        store_ckpt_head_fn: Per-head checkpoint writer `(s, v, w_idx, state)`
            when `state_prev` is `None`.
        sub_block_heads: Number of value heads kept live in registers at once
            on single-token decode (`chunk_size == 1`).

    Returns:
        `(out, states)` when `state_prev` is provided (with `states` shaped
        `[seq, window_size, num_v_heads, kq_head_dim, v_head_dim]`; positions
        `>= real_sizes` repeat the last valid state and are never written back
        to HBM), or `(out, None)` when streaming via callbacks.
    """
    if sub_block_heads <= 0:
        raise ValueError(f"sub_block_heads must be positive; got {sub_block_heads}")
    if not (1 <= cfg.window_size <= cfg.chunk_size):
        raise ValueError(
            "recurrent_gdn requires 1 <= cfg.window_size <= cfg.chunk_size; "
            f"got window_size={cfg.window_size}, chunk_size={cfg.chunk_size}."
        )

    if state_prev is not None:
        if load_head_fn is not None or store_ckpt_head_fn is not None:
            raise ValueError(
                "load_head_fn and store_ckpt_head_fn cannot be combined with "
                "state_prev."
            )
        load_head, store_ckpt, collect_states = _make_in_memory_state_io(
            state_prev, cfg
        )
    else:
        if load_head_fn is None or store_ckpt_head_fn is None:
            raise ValueError(
                "load_head_fn and store_ckpt_head_fn are required when "
                "state_prev is None."
            )
        load_head = load_head_fn
        store_ckpt = store_ckpt_head_fn
        collect_states = None

    steps = _prepare_recurrent_steps(
        real_sizes,
        q_compact,
        k_compact,
        v_compact,
        b_compact,
        a_compact,
        a_log,
        dt_bias,
        cfg,
    )
    if cfg.chunk_size == 1:
        out_arr = _run_single_token_recurrent(
            steps[0], load_head, store_ckpt, cfg, sub_block_heads
        )
    else:
        out_arr = _run_multi_token_recurrent(
            steps, state_prev, load_head, store_ckpt, cfg
        )
    return out_arr, (None if collect_states is None else collect_states())
