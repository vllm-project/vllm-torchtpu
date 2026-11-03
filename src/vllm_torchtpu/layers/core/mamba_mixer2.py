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

from vllm_torchtpu.kernels.ssd_scan import ssd_minimal_discrete_hybrid
from vllm_torchtpu.layers.core import causal_conv1d
from vllm_torchtpu.layers.core.mamba_utils import (
    broadcast_groups_to_heads,
    jax_gated_rms_norm,
    pack_ragged_ssm_chunks,
    prepare_mamba_batch_metadata,
    run_replicated_shard_map,
    scatter_mamba_states,
    seed_chunk_initial_states,
)


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

    j_conv_weight_3d = (
        jnp.expand_dims(j_conv_weight, axis=1)
        if j_conv_weight.ndim == 2
        else j_conv_weight
    )
    out_conv, new_conv_state = causal_conv1d.ragged_causal_conv1d(
        hidden_states_B_C,
        conv_state,
        j_conv_weight_3d,
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
    B = broadcast_groups_to_heads(
        B_raw.reshape(num_tokens, n_groups, d_state), num_heads
    )
    C = broadcast_groups_to_heads(
        C_raw.reshape(num_tokens, n_groups, d_state), num_heads
    )

    dt = jax.nn.softplus(dt_raw.astype(jnp.float32) + j_dt_bias.astype(jnp.float32))
    decay = dt * j_A.astype(jnp.float32)

    def _decode_ssm() -> tuple[jax.Array, jax.Array]:
        k = min(num_tokens, max_reqs)
        if k == 0:
            return (
                recurrent_state,
                jnp.zeros(
                    (num_tokens, num_heads * head_dim),
                    dtype=j_projected_states.dtype,
                ),
            )

        valid_k = meta.active_nonempty_seq[:k]
        need_init = meta.valid_state_slot[:k] & meta.has_initial_state[:k]
        init_h = jnp.where(
            need_init[:, None, None, None],
            recurrent_state[meta.safe_state_indices[:k]].astype(jnp.float32),
            0.0,
        )

        x_f32 = X[:k].astype(jnp.float32)
        b_f32 = B[:k].astype(jnp.float32)
        c_f32 = C[:k].astype(jnp.float32)
        dA = jnp.exp(decay[:k])
        x_dt_f32 = x_f32 * dt[:k, :, None]

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
        y_k = jnp.where(valid_k[:, None, None], y_k, 0.0)
        y_final = y_k.reshape(k, num_heads * head_dim).astype(j_projected_states.dtype)
        if num_tokens > k:
            y_final = jnp.pad(y_final, ((0, num_tokens - k), (0, 0)))

        updated_rec = scatter_mamba_states(
            recurrent_state, state_indices[:k], meta.valid_state_slot[:k], new_h
        )
        return updated_rec, y_final

    def _prefill_ssm() -> tuple[jax.Array, jax.Array]:
        X_dt = (X.astype(jnp.float32) * dt[..., None]).astype(X.dtype)
        (
            X_c,
            B_c,
            C_c,
            decay_c,
            reset_mask,
            last_chunk_indices,
            start_chunk_indices,
            padded_indices_valid,
        ) = pack_ragged_ssm_chunks(
            X_dt,
            B,
            C,
            decay,
            query_start_loc,
            distribution,
            chunk_size,
            compute_dtype=X.dtype,
        )

        need_init_seq = meta.valid_state_slot & meta.has_initial_state
        init_states_for_seqs = jnp.where(
            need_init_seq[:, None, None, None],
            recurrent_state[meta.safe_state_indices].astype(jnp.float32),
            0.0,
        )
        init_h_per_chunk = seed_chunk_initial_states(
            init_states_for_seqs, start_chunk_indices, X_c.shape[0]
        )

        Y_c, new_recurrent_states = ssd_minimal_discrete_hybrid(
            X_c,
            decay_c,
            B_c,
            C_c,
            last_chunk_indices,
            init_h_per_chunk,
            reset_mask,
            interpret=interpret,
        )

        Y_flat = Y_c.transpose(0, 2, 1, 3).reshape(-1, num_heads * head_dim)
        y_final = (
            Y_flat[padded_indices_valid].astype(jnp.float32)
            + X.reshape(num_tokens, num_heads * head_dim).astype(jnp.float32)
            * jnp.repeat(j_D.astype(jnp.float32), head_dim)[None, :]
        ).astype(j_projected_states.dtype)

        updated_rec = scatter_mamba_states(
            recurrent_state, state_indices, meta.valid_state_slot, new_recurrent_states
        )
        return updated_rec, y_final

    updated_recurrent_state, Y_final = jax.lax.cond(
        meta.is_pure_decode, _decode_ssm, _prefill_ssm
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
    chunk_size: int = 64,
    rms_norm_eps: float = 1e-5,
    apply_rms_norm: bool = True,
    has_silu_activation: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run Mamba-2 mixer over the worker's single-device TPU mesh."""
    if j_conv_weight.ndim == 2:
        j_conv_weight = jnp.expand_dims(j_conv_weight, axis=1)

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
