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

"""GDN/KDA verify-window recurrence, one window position and value head at a time."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import (
    compute_gdn,
    compute_kda,
    config,
    memory_ref,
)


def _l2_norm(x: jax.Array, *, is_kda: bool, eps: float = 1e-6) -> jax.Array:
    """L2-normalizes along the last axis using KDA rsqrt or GDN sqrt."""
    if not is_kda:
        return compute_gdn.l2_norm(x, eps=eps)
    x_f32 = x.astype(jnp.float32)
    inv_norm = jax.lax.rsqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 * inv_norm).astype(x.dtype)


def run(
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size], post-activation
    b_slot_ref: jax.Ref,  # [seq, chunk, 1, aligned_num_v_heads]
    a_slot_ref: jax.Ref,  # [seq, chunk, 1, aligned_gate_dim]
    rec_in_ref: jax.Ref,  # [seq, 1, num_v_heads, kq // packing, v] uint32
    out_slot_ref: jax.Ref,  # [seq, chunk, num_v_heads, v]
    rec_out_ref: jax.Ref,  # [seq, window, num_v_heads, kq // packing, v] uint32
    metadata_ref: memory_ref.MetadataRef,
    a_log: jax.Array,  # [aligned_num_v_heads] or [num_v_heads]
    dt_bias: jax.Array,  # [aligned_num_v_heads] or [num_v_heads, kq_head_dim]
    state_ref: jax.Ref,  # [seq, num_v_heads, kq, v] float32
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array,
) -> None:
    """Same arithmetic as recurrent_gdn / recurrent_kda, per position and head.

    Args:
        qkv_slot_ref: Post-activation QKV VMEM slot reference.
        b_slot_ref: Beta gate input VMEM slot reference.
        a_slot_ref: Decay gate input VMEM slot reference.
        rec_in_ref: Initial recurrent state VMEM slot reference.
        out_slot_ref: Output activation VMEM slot reference.
        rec_out_ref: Output recurrent state checkpoint VMEM slot reference.
        metadata_ref: Metadata reference for tile sequence records.
        a_log: Log-decay weight array.
        dt_bias: Timestep bias weight array.
        state_ref: Per-tile float32 recurrent state carry scratch reference.
        cfg: GDN/KDA kernel configuration.
        p_id: Current pipeline tile program ID.
    """
    compute = cfg.dtypes.compute
    state_dtype = cfg.dtypes.recurrent_state
    num_kq_heads = cfg.num_kq_heads
    kq_head_dim = cfg.kq_head_dim
    v_head_dim = cfg.v_head_dim
    k_offset = num_kq_heads * kq_head_dim
    v_offset = 2 * num_kq_heads * kq_head_dim
    is_kda = cfg.is_kda
    dot_dims = (((2,), (1,)), ((0,), (0,)))
    if state_ref is None:
        raise ValueError("state_ref must be allocated when window_tile is on.")

    real_sizes_list = []
    has_init_by_idx = []
    for idx in range(cfg.seq_tile_size):
        s_idx = metadata_ref.p_id_to_s_idx[p_id, idx]
        real_sizes_list.append(metadata_ref.p_id_to_r_size[p_id, idx])
        has_init_by_idx.append(metadata_ref.s_idx_has_initial_state[s_idx])
    real_sizes = jnp.stack(real_sizes_list, axis=0)

    q_compact = jnp.stack(
        [
            qkv_slot_ref[..., h * kq_head_dim : (h + 1) * kq_head_dim]
            for h in range(num_kq_heads)
        ],
        axis=1,
    )
    k_compact = jnp.stack(
        [
            qkv_slot_ref[
                ...,
                k_offset + h * kq_head_dim : k_offset + (h + 1) * kq_head_dim,
            ]
            for h in range(num_kq_heads)
        ],
        axis=1,
    )
    v_compact = jnp.stack(
        [
            qkv_slot_ref[
                ...,
                v_offset + h * v_head_dim : v_offset + (h + 1) * v_head_dim,
            ]
            for h in range(cfg.num_v_heads)
        ],
        axis=1,
    )
    b_compact = jnp.expand_dims(b_slot_ref[...], axis=1)

    mask_dtype = compute_gdn.get_mask_dtype(compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2
    )
    mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

    q = _l2_norm(jnp.where(mask, q_compact.astype(compute), 0), is_kda=is_kda) * (
        kq_head_dim**-0.5
    )
    k = _l2_norm(jnp.where(mask, k_compact.astype(compute), 0), is_kda=is_kda)
    v = jnp.where(mask, v_compact.astype(compute), 0)
    qk_dot = jnp.sum(q * k, axis=-1, keepdims=True)

    beta = jnp.where(mask, jax.nn.sigmoid(b_compact.astype(compute)), 0)
    beta = compute_gdn.fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)[
        :, : cfg.num_v_heads
    ]

    if is_kda:
        a_compact = jnp.stack(
            [
                a_slot_ref[..., h * kq_head_dim : (h + 1) * kq_head_dim]
                for h in range(cfg.num_v_heads)
            ],
            axis=1,
        )
        a_log_kda = a_log.reshape(1, cfg.num_v_heads, 1, 1, 1).astype(compute)
        dt_bias_kda = dt_bias.reshape(1, cfg.num_v_heads, 1, 1, kq_head_dim).astype(
            compute
        )
        gating_log = jnp.where(
            mask,
            compute_kda.activate_gate(a_compact, a_log_kda, dt_bias_kda, cfg),
            0.0,
        )
        gating_decay = jnp.exp(gating_log)[:, :, :, 0, :]
    else:
        a_compact = jnp.expand_dims(a_slot_ref[...], axis=1)
        a_log_gdn = a_log.reshape(1, 1, 1, 1, -1).astype(compute)
        dt_bias_gdn = dt_bias.reshape(1, 1, 1, 1, -1).astype(compute)
        gating_log = jnp.where(
            mask,
            -jnp.exp(a_log_gdn)
            * jax.nn.softplus(a_compact.astype(compute) + dt_bias_gdn),
            0,
        )
        gating_decay = compute_gdn.fused_transpose_broadcast(
            jnp.exp(gating_log), src_dim=4, dst_dim=1
        )[:, : cfg.num_v_heads]

    seqs_per_group = max(1, 64 // cfg.num_v_heads)
    for idx_start in range(0, cfg.seq_tile_size, seqs_per_group):
        idx_end = min(cfg.seq_tile_size, idx_start + seqs_per_group)
        group = []
        for idx in range(idx_start, idx_end):
            for kq_head in range(num_kq_heads):
                for head in range(
                    kq_head * cfg.v_per_kq_head,
                    (kq_head + 1) * cfg.v_per_kq_head,
                ):
                    group.append((idx, kq_head, head))
        states = [
            jnp.where(
                has_init_by_idx[idx],
                pltpu.bitcast(rec_in_ref[idx, 0, head], state_dtype).astype(
                    jnp.float32
                ),
                0,
            )
            for idx, _, head in group
        ]
        for pos in range(cfg.chunk_size):
            decayed_list = []
            kq_list = []
            kq_cache = {}
            for state, (idx, kq_head, head) in zip(states, group):
                decay = (
                    gating_decay[idx, head, pos, :, None]
                    if is_kda
                    else gating_decay[idx, head, pos]
                )
                decayed_list.append(state * decay)
                kq_key = (idx, kq_head)
                if kq_key not in kq_cache:
                    kq_cache[kq_key] = jnp.concat(
                        [k[idx, kq_head, pos], q[idx, kq_head, pos]], axis=0
                    )
                kq_list.append(kq_cache[kq_key])

            kq_projs = [
                jax.lax.dot_general(
                    kq_item[None],
                    decayed[None],
                    dimension_numbers=dot_dims,
                    preferred_element_type=jnp.float32,
                )[0].astype(compute)
                for kq_item, decayed in zip(kq_list, decayed_list)
            ]

            v_news = [
                beta[idx, head, pos] * (v[idx, head, pos] - kq_proj[:1])
                for kq_proj, (idx, _, head) in zip(kq_projs, group)
            ]

            out_rows_pos = {idx: [] for idx in range(idx_start, idx_end)}
            for kq_proj, v_new, (idx, kq_head, _) in zip(kq_projs, v_news, group):
                out_rows_pos[idx].append(
                    (kq_proj[1:] + qk_dot[idx, kq_head, pos] * v_new).astype(compute)
                )
            for idx in range(idx_start, idx_end):
                out_slot_ref[idx, pos] = jnp.concat(out_rows_pos[idx], axis=0).astype(
                    out_slot_ref.dtype
                )

            kt_cache = {
                (idx, kq_head): compute_gdn.fused_transpose_broadcast(
                    k[idx, kq_head, pos], src_dim=1, dst_dim=0
                )
                for idx, kq_head in kq_cache
            }

            states = [
                decayed + kt_cache[(idx, kq_head)] * v_new
                for decayed, v_new, (idx, kq_head, _) in zip(
                    decayed_list, v_news, group
                )
            ]

            for state, (idx, _, head) in zip(states, group):
                rec_out_ref[idx, pos, head] = pltpu.bitcast(
                    state.astype(state_dtype), jnp.uint32
                )
