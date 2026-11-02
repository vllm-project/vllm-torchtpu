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

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.causal_conv1d import (
    causal_conv1d,  # pyrefly: ignore[missing-module-attribute]
)
from vllm_torchtpu.layers.core.ssd_scan import ssd_minimal_discrete_hybrid

_HEAD_SHARDING_AXIS = "model"


def broadcast_groups_to_heads(x: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    """Broadcasts group dimension to head dimension.

    x: (..., n_groups, dim) -> (..., n_heads, dim)
    """
    n_groups = x.shape[-2]
    if n_groups == n_heads:
        return x
    group_size = n_heads // n_groups
    return jnp.repeat(x, group_size, axis=-2)


def jax_gated_rms_norm(
    x: jnp.ndarray,
    gate: jnp.ndarray,
    weight: jnp.ndarray | None,
    eps: float,
    n_groups: int = 1,
) -> jnp.ndarray:
    input_dtype = x.dtype

    # 1. Apply Gate (SiLU)
    x_gated = x.astype(jnp.float32) * jax.nn.silu(gate.astype(jnp.float32))
    if weight is None:
        return x_gated.astype(input_dtype)

    # 2. Compute Variance and Normalize
    orig_shape = x_gated.shape
    if n_groups > 1:
        x_gated = x_gated.reshape(
            *orig_shape[:-1], n_groups, orig_shape[-1] // n_groups
        )
    variance = jnp.mean(jnp.square(x_gated), axis=-1, keepdims=True)
    x_norm = (x_gated * jax.lax.rsqrt(variance + eps)).reshape(orig_shape)

    # 3. Apply weight
    return (weight.astype(jnp.float32) * x_norm).astype(input_dtype)


def pack_inputs_mamba(
    X: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    decay: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    chunk_size: int,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Pads each sequence to a multiple of chunk_size and packs into a stream."""
    num_tokens = X.shape[0]
    num_seqs = query_start_loc.shape[0] - 1

    num_valid_seqs = jnp.clip(distribution[2], 0, num_seqs)
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(
        valid_loc_mask, query_start_loc, last_valid_loc
    )

    # Calculate sequence lengths and pad them to multiples of chunk_size.
    seq_lengths = jnp.maximum(
        effective_query_start_loc[1:] - effective_query_start_loc[:-1], 0
    )
    num_chunks = (seq_lengths + chunk_size - 1) // chunk_size
    padded_lengths = num_chunks * chunk_size

    new_query_start_loc = jnp.cumsum(
        jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), padded_lengths])
    )

    # Map each original token index to its sequence ID.
    token_indices = jnp.arange(num_tokens, dtype=jnp.int32)
    valid_token_mask = token_indices < last_valid_loc
    seq_id = (
        jnp.searchsorted(effective_query_start_loc, token_indices, side="right") - 1
    )
    seq_id = jnp.clip(seq_id, 0, num_seqs - 1)
    original_start = effective_query_start_loc[seq_id]
    new_start = new_query_start_loc[seq_id]

    max_packed_tokens = num_tokens + num_seqs * chunk_size
    max_packed_tokens = (max_packed_tokens + chunk_size - 1) // chunk_size * chunk_size

    padded_indices_valid = jnp.clip(
        new_start + (token_indices - original_start), 0, max_packed_tokens - 1
    )
    scatter_token_indices = jnp.where(
        valid_token_mask, padded_indices_valid, max_packed_tokens
    )

    # Flatten last dimensions to concatenate for a single scatter.
    n_heads = X.shape[1]
    d_head = X.shape[2]
    d_state = B.shape[2]

    X_flat = X.reshape(num_tokens, n_heads * d_head)
    B_flat = B.reshape(num_tokens, n_heads * d_state)
    C_flat = C.reshape(num_tokens, n_heads * d_state)

    combined = jnp.concatenate(
        [
            X_flat.astype(compute_dtype),
            B_flat.astype(compute_dtype),
            C_flat.astype(compute_dtype),
        ],
        axis=-1,
    )

    output_shape = (max_packed_tokens,) + combined.shape[1:]
    packed_combined = jnp.zeros(output_shape, dtype=compute_dtype)
    packed_combined = packed_combined.at[scatter_token_indices].set(
        combined, mode="drop"
    )

    # Split back.
    dim_X = n_heads * d_head
    dim_B = n_heads * d_state

    packed_X_flat = packed_combined[..., :dim_X]
    packed_B_flat = packed_combined[..., dim_X : dim_X + dim_B]
    packed_C_flat = packed_combined[..., dim_X + dim_B :]

    num_chunks_total = max_packed_tokens // chunk_size

    X_c = packed_X_flat.reshape(
        num_chunks_total, chunk_size, n_heads, d_head
    ).transpose(0, 2, 1, 3)
    B_c = packed_B_flat.reshape(
        num_chunks_total, chunk_size, n_heads, d_state
    ).transpose(0, 2, 1, 3)
    C_c = packed_C_flat.reshape(
        num_chunks_total, chunk_size, n_heads, d_state
    ).transpose(0, 2, 1, 3)

    # For decay (float32, shape (num_tokens, n_heads)).
    output_shape_f32 = (max_packed_tokens, n_heads)
    packed_decay = jnp.zeros(output_shape_f32, dtype=jnp.float32)
    packed_decay = packed_decay.at[scatter_token_indices].set(
        decay.astype(jnp.float32), mode="drop"
    )

    decay_c = packed_decay.reshape(num_chunks_total, chunk_size, n_heads).transpose(
        0, 2, 1
    )

    # Map each packed chunk directly to its owning sequence index via 1D
    # searchsorted so prefill can gather initial states once from recurrent_state.
    seq_chunk_starts = new_query_start_loc // chunk_size
    chunk_ids = jnp.arange(num_chunks_total, dtype=jnp.int32)
    valid_chunk = chunk_ids < seq_chunk_starts[-1]
    chunk_seq_idx = jnp.clip(
        jnp.searchsorted(seq_chunk_starts, chunk_ids, side="right") - 1,
        0,
        num_seqs - 1,
    )
    reset_mask = valid_chunk & (chunk_ids == seq_chunk_starts[:-1][chunk_seq_idx])

    last_chunk_indices = jnp.clip(seq_chunk_starts[1:] - 1, 0, num_chunks_total - 1)

    return (
        X_c,
        B_c,
        C_c,
        decay_c,
        reset_mask,
        last_chunk_indices,
        chunk_seq_idx,
        padded_indices_valid,
    )


def run_jax_mamba_mixer_local(
    j_projected_states: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: jnp.ndarray | None,
    j_A: jnp.ndarray,
    j_D: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    j_norm_weight: jnp.ndarray | None,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
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
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the local device execution of Mamba-2 Mixer."""
    num_tokens = j_projected_states.shape[0]
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

    max_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1 : max_reqs + 1] - query_start_loc[:max_reqs]
    has_initial_state = (seq_lens - query_lens) > 0

    j_conv_weight_local = j_conv_weight
    if j_conv_weight_local.ndim == 2:
        j_conv_weight_local = jnp.expand_dims(j_conv_weight_local, axis=1)

    out_conv, new_conv_state = causal_conv1d.ragged_causal_conv1d(
        hidden_states_B_C,
        conv_state,
        j_conv_weight_local,
        j_conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )
    if has_silu_activation:
        out_conv = jax.nn.silu(out_conv)

    local_groups_ssm = n_groups * d_state

    X_raw, B_raw, C_raw = jnp.split(
        out_conv,
        [
            tped_intermediate_size,
            tped_intermediate_size + local_groups_ssm,
        ],
        axis=-1,
    )

    X = X_raw.reshape(num_tokens, num_heads, head_dim)
    B_raw = B_raw.reshape(num_tokens, n_groups, d_state)
    C_raw = C_raw.reshape(num_tokens, n_groups, d_state)

    B = broadcast_groups_to_heads(B_raw, num_heads)
    C = broadcast_groups_to_heads(C_raw, num_heads)

    dt = jax.nn.softplus(dt_raw.astype(jnp.float32) + j_dt_bias.astype(jnp.float32))
    decay = dt * j_A.astype(jnp.float32)

    num_slots = recurrent_state.shape[0]
    num_valid_seqs = jnp.clip(distribution[2], 0, max_reqs)
    seq_arange = jnp.arange(max_reqs, dtype=jnp.int32)
    active_seq_mask = seq_arange < num_valid_seqs
    is_pure_decode = (
        (distribution[0] == distribution[2])
        & (num_valid_seqs <= num_tokens)
        & jnp.all(jnp.where(active_seq_mask, query_lens == 1, True))
    )

    def _decode_ssm() -> tuple[jnp.ndarray, jnp.ndarray]:
        k = min(num_tokens, max_reqs)
        if k == 0:
            return (
                recurrent_state,
                jnp.zeros(
                    (num_tokens, num_heads * head_dim),
                    dtype=j_projected_states.dtype,
                ),
            )

        req_arange = jnp.arange(k, dtype=jnp.int32)
        valid_k = (req_arange < num_valid_seqs) & (query_lens[:k] > 0)
        need_init = valid_k & has_initial_state[:k]
        safe_read_slots = jnp.where(
            need_init, jnp.clip(state_indices[:k], 0, num_slots - 1), 0
        )
        init_h = jnp.where(
            need_init[:, None, None, None],
            recurrent_state[safe_read_slots].astype(jnp.float32),
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

        valid_write = (
            valid_k & (state_indices[:k] >= 0) & (state_indices[:k] < num_slots)
        )
        scatter_slots = jnp.where(valid_write, state_indices[:k], num_slots)
        updated_rec = recurrent_state.at[scatter_slots].set(
            new_h.astype(recurrent_state.dtype), mode="drop"
        )
        return updated_rec, y_final

    def _prefill_ssm() -> tuple[jnp.ndarray, jnp.ndarray]:
        X_dt = (X.astype(jnp.float32) * dt[..., None]).astype(X.dtype)
        (
            X_c,
            B_c,
            C_c,
            decay_c,
            reset_mask,
            last_chunk_indices,
            chunk_seq_idx,
            padded_indices_valid,
        ) = pack_inputs_mamba(
            X_dt,
            B,
            C,
            decay,
            query_start_loc,
            distribution,
            chunk_size,
            compute_dtype=X.dtype,
        )

        chunk_needs_init = (
            reset_mask
            & (chunk_seq_idx < num_valid_seqs)
            & has_initial_state[chunk_seq_idx]
            & (query_lens[chunk_seq_idx] > 0)
        )
        safe_chunk_slots = jnp.where(
            chunk_needs_init,
            jnp.clip(state_indices[chunk_seq_idx], 0, num_slots - 1),
            0,
        )
        init_h_per_chunk = jnp.where(
            chunk_needs_init[:, None, None, None],
            recurrent_state[safe_chunk_slots].astype(jnp.float32),
            0.0,
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

        valid_seq_mask = (
            active_seq_mask
            & (query_lens > 0)
            & (state_indices >= 0)
            & (state_indices < num_slots)
        )
        scatter_state_indices = jnp.where(valid_seq_mask, state_indices, num_slots)
        updated_rec = recurrent_state.at[scatter_state_indices].set(
            new_recurrent_states.astype(recurrent_state.dtype), mode="drop"
        )
        return updated_rec, y_final

    updated_recurrent_state, Y_final = jax.lax.cond(
        is_pure_decode, _decode_ssm, _prefill_ssm
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

    return (new_conv_state, updated_recurrent_state), Y_out


def run_jax_mamba_mixer(
    j_projected_states: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: jnp.ndarray | None,
    j_A: jnp.ndarray,
    j_D: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    j_norm_weight: jnp.ndarray | None,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
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
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the sharded JAX Mamba-2 Mixer over the given TPU mesh."""
    if j_conv_weight.ndim == 2:
        j_conv_weight = jnp.expand_dims(j_conv_weight, axis=1)

    head_axis = _HEAD_SHARDING_AXIS if _HEAD_SHARDING_AXIS in mesh.axis_names else None

    conv_bias_spec = P(head_axis) if j_conv_bias is not None else None
    norm_weight_spec = P(head_axis) if j_norm_weight is not None else None

    in_specs = (
        P(None, head_axis),  # j_projected_states
        P(None, None, head_axis),  # conv_state
        P(None, head_axis, None, None),  # recurrent_state
        P(head_axis, None, None),  # j_conv_weight
        conv_bias_spec,  # j_conv_bias
        P(head_axis),  # j_A
        P(head_axis),  # j_D
        P(head_axis),  # j_dt_bias
        norm_weight_spec,  # j_norm_weight
        P(None),  # state_indices
        P(None),  # query_start_loc
        P(None),  # distribution
        P(None),  # seq_lens
    )

    out_specs = (
        (
            P(None, None, head_axis),  # new_conv_state
            P(None, head_axis, None, None),  # updated_recurrent_state
        ),
        P(None, head_axis),  # Y_out
    )

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

    mapped_fn = jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    return mapped_fn(
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
