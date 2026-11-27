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
"""JAX/Pallas implementation of Mamba-2 Mixer core for TPU."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu.layers.core import causal_conv1d
from vllm_torchtpu.layers.core.mamba_utils import (
    TPU_SSD_BLOCK_LEN,
    broadcast_groups_to_heads,
    gather_active_mamba_states,
    jax_gated_rms_norm,
    pack_active_chunk_initial_states,
    pack_ragged_ssm_chunks,
    prepare_mamba_batch_metadata,
    run_replicated_shard_map,
    scatter_mamba_states,
)
from vllm_torchtpu.layers.core.ssd_scan import ssd_minimal_discrete_hybrid


def run_jax_mamba_mixer_local(
    j_projected_states: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    j_conv_weight: jax.Array,
    j_conv_bias: jax.Array | None,
    j_A: jax.Array,
    j_D: jax.Array,
    j_dt_bias: jax.Array,
    j_norm_weight: jax.Array | None,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    tped_intermediate_size: int,
    tped_conv_size: int,
    tped_dt_size: int,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    kernel_size: int,
    chunk_size: int,
    rms_norm_eps: float,
    apply_rms_norm: bool = True,
    has_silu_activation: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run local per-rank Mamba-2 mixer over a ragged batch."""
    num_tokens = j_projected_states.shape[0]
    max_reqs = seq_lens.shape[0]
    num_slots = recurrent_state.shape[0]

    X_MLP_local = j_projected_states[..., :tped_intermediate_size]
    hidden_states_B_C = j_projected_states[
        ...,
        tped_intermediate_size : tped_intermediate_size + tped_conv_size,
    ]
    dt_raw = j_projected_states[
        ...,
        tped_intermediate_size + tped_conv_size : tped_intermediate_size
        + tped_conv_size
        + tped_dt_size,
    ]

    meta = prepare_mamba_batch_metadata(
        query_start_loc,
        seq_lens,
        distribution,
        state_indices,
        num_tokens=num_tokens,
        num_slots=num_slots,
    )

    out_conv, new_conv_state = causal_conv1d.ragged_causal_conv1d(
        hidden_states_B_C,
        conv_state,
        j_conv_weight,
        j_conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        meta.has_initial_state,
        activation="silu" if has_silu_activation else None,
        kernel_size=kernel_size,
    )

    local_groups_ssm = n_groups * d_state
    X_raw, B_raw, C_raw = jnp.split(
        out_conv,
        [tped_intermediate_size, tped_intermediate_size + local_groups_ssm],
        axis=-1,
    )

    X = X_raw.reshape(num_tokens, num_heads, head_dim)
    B_grouped = B_raw.reshape(num_tokens, n_groups, d_state)
    C_grouped = C_raw.reshape(num_tokens, n_groups, d_state)

    dt = jax.nn.softplus(dt_raw.astype(jnp.float32) + j_dt_bias.astype(jnp.float32))
    decay = dt * j_A.astype(jnp.float32)

    compact_ids = meta.compact_seq_ids
    k_cap = compact_ids.shape[0]
    compact_slots = state_indices[compact_ids]
    compact_safe_slots = meta.safe_state_indices[compact_ids]
    compact_valid_slot = meta.compact_valid & meta.valid_state_slot[compact_ids]
    compact_need_init = compact_valid_slot & meta.has_initial_state[compact_ids]

    def _decode_ssm() -> tuple[jax.Array, jax.Array]:
        if num_tokens == 0:
            return (
                jnp.zeros((k_cap, num_heads, head_dim, d_state), dtype=jnp.float32),
                jnp.zeros(
                    (num_tokens, num_heads * head_dim),
                    dtype=j_projected_states.dtype,
                ),
            )

        tok_pos = jnp.clip(
            meta.effective_query_start_loc[compact_ids],
            0,
            jnp.maximum(num_tokens - 1, 0),
        )
        init_h = gather_active_mamba_states(
            recurrent_state,
            compact_safe_slots,
            compact_need_init,
            num_active=meta.num_nonempty_seqs,
        )

        x_f32 = X[tok_pos].astype(jnp.float32)
        b_f32 = broadcast_groups_to_heads(B_grouped[tok_pos], num_heads).astype(
            jnp.float32
        )
        c_f32 = broadcast_groups_to_heads(C_grouped[tok_pos], num_heads).astype(
            jnp.float32
        )
        dA = jnp.exp(decay[tok_pos])
        x_dt_f32 = x_f32 * dt[tok_pos, :, None]

        new_h = (
            init_h * dA[:, :, None, None]
            + x_dt_f32[:, :, :, None] * b_f32[:, :, None, :]
        )
        # Preserve float32 state carry during decode readout contraction.
        y_k = (
            jnp.einsum(
                "khpn,khn->khp",
                new_h,
                c_f32,
                precision=jax.lax.Precision.HIGHEST,
            )
            + x_f32 * j_D.astype(jnp.float32)[None, :, None]
        )
        y_k = y_k.reshape(k_cap, num_heads * head_dim).astype(j_projected_states.dtype)
        dst_tok = jnp.where(meta.compact_valid, tok_pos, num_tokens)
        y_final = (
            jnp.zeros(
                (num_tokens, num_heads * head_dim), dtype=j_projected_states.dtype
            )
            .at[dst_tok]
            .set(y_k, mode="drop")
        )
        return new_h, y_final

    def _prefill_ssm() -> tuple[jax.Array, jax.Array]:
        X_dt = (X.astype(jnp.float32) * dt[..., None]).astype(X.dtype)
        (
            X_c,
            B_c,
            C_c,
            decay_c,
            reset_mask,
            last_chunk_indices,
            active_chunk_seq,
            padded_indices_valid,
        ) = pack_ragged_ssm_chunks(
            X_dt,
            B_grouped,
            C_grouped,
            decay,
            query_start_loc,
            distribution,
            chunk_size,
            compute_dtype=X.dtype,
        )

        active_chunk_compact = meta.compact_valid & active_chunk_seq[compact_ids]
        should_read_init = compact_need_init & active_chunk_compact
        init_states_compact = gather_active_mamba_states(
            recurrent_state,
            compact_safe_slots,
            should_read_init,
            num_active=meta.num_nonempty_seqs,
        )
        init_h_per_seq = pack_active_chunk_initial_states(
            init_states_compact,
            active_chunk_compact,
            jnp.any(should_read_init),
            num_active=meta.num_nonempty_seqs,
            max_chunk_seqs=min(max_reqs, max(1, num_tokens)),
        )

        Y_c, new_recurrent_states = ssd_minimal_discrete_hybrid(
            X_c,
            decay_c,
            B_c,
            C_c,
            last_chunk_indices[compact_ids],
            init_h_per_seq,
            reset_mask,
            interpret=interpret,
        )

        valid_c = padded_indices_valid // chunk_size
        valid_l = padded_indices_valid % chunk_size
        y_scan_tokens = (
            Y_c[valid_c, :, valid_l, :]
            .reshape(num_tokens, num_heads * head_dim)
            .astype(jnp.float32)
        )
        y_final = (
            y_scan_tokens
            + X.reshape(num_tokens, num_heads * head_dim).astype(jnp.float32)
            * jnp.repeat(j_D.astype(jnp.float32), head_dim)[None, :]
        ).astype(j_projected_states.dtype)
        return new_recurrent_states, y_final

    new_states_compact, Y_final = jax.lax.cond(
        meta.is_pure_decode, _decode_ssm, _prefill_ssm
    )
    updated_recurrent_state = scatter_mamba_states(
        recurrent_state,
        compact_slots,
        compact_valid_slot,
        new_states_compact,
        num_active=meta.num_nonempty_seqs,
    )

    if apply_rms_norm:
        Y_out = jax_gated_rms_norm(
            x=Y_final,
            gate=X_MLP_local,
            weight=j_norm_weight,
            eps=rms_norm_eps,
            n_groups=n_groups,
        )
    else:
        Y_out = Y_final.astype(j_projected_states.dtype)

    return new_conv_state, updated_recurrent_state, Y_out


def run_jax_mamba_mixer(
    j_projected_states: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    j_conv_weight: jax.Array,
    j_conv_bias: jax.Array | None,
    j_A: jax.Array,
    j_D: jax.Array,
    j_dt_bias: jax.Array,
    j_norm_weight: jax.Array | None,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    tped_intermediate_size: int,
    tped_conv_size: int,
    tped_dt_size: int,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    kernel_size: int,
    chunk_size: int = TPU_SSD_BLOCK_LEN,
    rms_norm_eps: float = 1e-5,
    apply_rms_norm: bool = True,
    has_silu_activation: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run Mamba-2 mixer over the worker's single-device TPU mesh."""
    local_fn = functools.partial(
        run_jax_mamba_mixer_local,
        tped_intermediate_size=tped_intermediate_size,
        tped_conv_size=tped_conv_size,
        tped_dt_size=tped_dt_size,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        rms_norm_eps=rms_norm_eps,
        apply_rms_norm=apply_rms_norm,
        has_silu_activation=has_silu_activation,
        interpret=interpret,
    )
    return run_replicated_shard_map(
        local_fn,
        mesh,
        j_projected_states,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A,
        j_D,
        j_dt_bias,
        j_norm_weight,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
    )
