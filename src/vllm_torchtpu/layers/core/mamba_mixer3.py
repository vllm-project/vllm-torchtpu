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
"""JAX implementation of Mamba-3 Mixer core for TPU."""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core.mamba3_ssd_scan import (  # pyrefly: ignore
    ssd3_minimal_discrete_hybrid,
)

_HEAD_SHARDING_AXIS = "model"


def broadcast_groups_to_heads(x: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    """Broadcasts group dimension to head dimension on axis 1.

    Args:
      x: (num_tokens, n_groups, ...)
      n_heads: Target number of heads
    """
    n_groups = x.shape[1]
    if n_groups == n_heads:
        return x
    group_size = n_heads // n_groups
    return jnp.repeat(x, group_size, axis=1)


def jax_rms_norm_per_feature(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Applies RMSNorm over the last dimension (d_state) for B and C projections."""
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = (x_f32 * jax.lax.rsqrt(variance + eps)).astype(x.dtype)
    return x_norm * weight.astype(x.dtype)


def jax_gated_rms_norm_mamba3(
    x: jnp.ndarray,
    gate: jnp.ndarray | None,
    weight: jnp.ndarray | None,
    eps: float,
    n_groups: int = 1,
) -> jnp.ndarray:
    """Applies SiLU gating (for SISO) and optional RMSNorm."""
    input_dtype = x.dtype
    x_val = x.astype(jnp.float32)
    if gate is not None:
        x_val = x_val * jax.nn.silu(gate.astype(jnp.float32))
    if weight is None:
        return x_val.astype(input_dtype)

    orig_shape = x_val.shape
    if n_groups > 1:
        x_val = x_val.reshape(*orig_shape[:-1], n_groups, orig_shape[-1] // n_groups)
    variance = jnp.mean(jnp.square(x_val), axis=-1, keepdims=True)
    x_norm = (x_val * jax.lax.rsqrt(variance + eps)).reshape(orig_shape)

    return (weight.astype(jnp.float32) * x_norm).astype(input_dtype)


def _rotate_pairs_siso(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    num_rope_angles: int,
) -> jnp.ndarray:
    """Applies 2D rotary embedding to interleaved pairs (2i, 2i+1) for SISO."""
    num_tokens, num_heads, d_state = x.shape
    rot_dim = 2 * num_rope_angles
    x_rot_part = x[..., :rot_dim].reshape(num_tokens, num_heads, num_rope_angles, 2)
    x0 = x_rot_part[..., 0].astype(jnp.float32)
    x1 = x_rot_part[..., 1].astype(jnp.float32)
    xo0 = x0 * cos - x1 * sin
    xo1 = x0 * sin + x1 * cos
    rotated = jnp.stack([xo0, xo1], axis=-1).reshape(num_tokens, num_heads, rot_dim)
    rotated = rotated.astype(x.dtype)
    if rot_dim < d_state:
        return jnp.concatenate([rotated, x[..., rot_dim:]], axis=-1)
    return rotated


def _rotate_split_half_mimo(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    num_rope_angles: int,
) -> jnp.ndarray:
    """Applies 2D split-half rotary embedding across rank channels for MIMO."""
    half = x.shape[-1] // 2
    x0 = x[..., :num_rope_angles].astype(jnp.float32)
    x_mid = x[..., num_rope_angles:half]
    x1 = x[..., half : half + num_rope_angles].astype(jnp.float32)
    x_tail = x[..., half + num_rope_angles :]
    xo0 = (x0 * cos - x1 * sin).astype(x.dtype)
    xo1 = (x0 * sin + x1 * cos).astype(x.dtype)
    return jnp.concatenate([xo0, x_mid, xo1, x_tail], axis=-1)


def _chunked_cumsum_axis0(x: jnp.ndarray, block_size: int = 64) -> jnp.ndarray:
    """Computes jnp.cumsum(x, axis=0) using two-level blocking."""
    n = x.shape[0]
    if n <= block_size:
        return jnp.cumsum(x, axis=0)
    rem = n % block_size
    if rem != 0:
        pad_len = block_size - rem
        pad_width = ((0, pad_len),) + ((0, 0),) * (x.ndim - 1)
        x_padded = jnp.pad(x, pad_width)
    else:
        x_padded = x
    n_blocks = x_padded.shape[0] // block_size
    x_blocks = x_padded.reshape((n_blocks, block_size) + x.shape[1:])
    intra = jnp.cumsum(x_blocks, axis=1)
    block_sums = intra[:, -1]
    inter = jnp.cumsum(block_sums, axis=0)
    inter_shifted = jnp.concatenate([jnp.zeros_like(inter[:1]), inter[:-1]], axis=0)
    out_blocks = intra + jnp.expand_dims(inter_shifted, axis=1)
    return out_blocks.reshape((-1,) + x.shape[1:])[:n]


def pack_inputs_mamba3(
    X: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    decay: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    chunk_size: int,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    skip_single_token_seqs: bool = False,
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
    """Pads each sequence to a multiple of chunk_size and packs into chunk tensors.

    Supports both SISO (X: (T, H, P), B/C: (T, H, N)) and MIMO
    (X: (T, H, R, P), B/C: (T, H, R, N), where R channels are flattened into
    block_len = chunk_size * R). When skip_single_token_seqs is True, 1-token
    decode sequences in a mixed batch allocate 0 chunks in the packed buffer.
    """
    num_tokens = X.shape[0]
    num_seqs = len(query_start_loc) - 1

    num_valid_seqs = distribution[2]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(
        valid_loc_mask, query_start_loc, last_valid_loc
    )

    seq_lengths = effective_query_start_loc[1:] - effective_query_start_loc[:-1]
    if skip_single_token_seqs:
        effective_seq_lengths = jnp.where(seq_lengths > 1, seq_lengths, 0)
    else:
        effective_seq_lengths = seq_lengths

    num_chunks = (effective_seq_lengths + chunk_size - 1) // chunk_size
    padded_lengths = num_chunks * chunk_size

    new_query_start_loc = jnp.cumsum(
        jnp.concatenate([jnp.array([0], dtype=jnp.int32), padded_lengths])
    )

    token_indices = jnp.arange(num_tokens)
    seq_id = (
        jnp.searchsorted(effective_query_start_loc, token_indices, side="right") - 1
    )
    seq_id = jnp.clip(seq_id, 0, num_seqs - 1)
    valid_token_mask = (token_indices < last_valid_loc) & (
        effective_seq_lengths[seq_id] > 0
    )
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

    n_heads = X.shape[1]
    is_mimo = X.ndim == 4
    mimo_rank = X.shape[2] if is_mimo else 1
    d_head = X.shape[-1]
    d_state = B.shape[-1]

    dim_X = n_heads * mimo_rank * d_head
    dim_B = n_heads * mimo_rank * d_state

    X_flat = X.reshape(num_tokens, dim_X)
    B_flat = B.reshape(num_tokens, dim_B)
    C_flat = C.reshape(num_tokens, dim_B)

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

    packed_X_flat = packed_combined[..., :dim_X]
    packed_B_flat = packed_combined[..., dim_X : dim_X + dim_B]
    packed_C_flat = packed_combined[..., dim_X + dim_B :]

    num_chunks_total = max_packed_tokens // chunk_size
    block_len = chunk_size * mimo_rank

    if is_mimo:
        X_c = (
            packed_X_flat.reshape(
                num_chunks_total, chunk_size, n_heads, mimo_rank, d_head
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(num_chunks_total, n_heads, block_len, d_head)
        )
        B_c = (
            packed_B_flat.reshape(
                num_chunks_total, chunk_size, n_heads, mimo_rank, d_state
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(num_chunks_total, n_heads, block_len, d_state)
        )
        C_c = (
            packed_C_flat.reshape(
                num_chunks_total, chunk_size, n_heads, mimo_rank, d_state
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(num_chunks_total, n_heads, block_len, d_state)
        )
    else:
        X_c = packed_X_flat.reshape(
            num_chunks_total, chunk_size, n_heads, d_head
        ).transpose(0, 2, 1, 3)
        B_c = packed_B_flat.reshape(
            num_chunks_total, chunk_size, n_heads, d_state
        ).transpose(0, 2, 1, 3)
        C_c = packed_C_flat.reshape(
            num_chunks_total, chunk_size, n_heads, d_state
        ).transpose(0, 2, 1, 3)

    output_shape_f32 = (max_packed_tokens, n_heads)
    packed_decay = jnp.zeros(output_shape_f32, dtype=jnp.float32)
    packed_decay = packed_decay.at[scatter_token_indices].set(
        decay.astype(jnp.float32), mode="drop"
    )

    decay_c = packed_decay.reshape(num_chunks_total, chunk_size, n_heads).transpose(
        0, 2, 1
    )

    active_chunk_seq = (jnp.arange(num_seqs) < num_valid_seqs) & (
        effective_seq_lengths > 0
    )
    start_chunk_indices = jnp.where(
        active_chunk_seq,
        new_query_start_loc[:-1] // chunk_size,
        num_chunks_total,
    )
    reset_mask = (
        jnp.zeros((num_chunks_total,), dtype=bool)
        .at[start_chunk_indices]
        .set(True, mode="drop")
    )

    last_chunk_indices = jnp.clip(
        (new_query_start_loc[1:] // chunk_size) - 1, 0, num_chunks_total - 1
    )

    return (
        X_c,
        B_c,
        C_c,
        decay_c,
        reset_mask,
        last_chunk_indices,
        start_chunk_indices,
        padded_indices_valid,
    )


def _project_and_norm_qkv(
    j_projected_states: jnp.ndarray,
    j_B_norm_weight: jnp.ndarray,
    j_C_norm_weight: jnp.ndarray,
    j_B_bias: jnp.ndarray,
    j_C_bias: jnp.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    r: int,
    is_mimo: bool,
    rms_norm_eps: float,
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
    """Splits projected states and computes RMS-normalized Q/K pre-rotations."""
    num_tokens = j_projected_states.shape[0]
    local_intermediate = num_heads * head_dim
    local_bc = r * n_groups * d_state

    s1 = local_intermediate
    s2 = s1 + local_intermediate
    s3 = s2 + local_bc
    s4 = s3 + local_bc
    s5 = s4 + num_heads
    s6 = s5 + num_heads
    s7 = s6 + num_heads
    (
        z_raw,
        v_raw,
        B_raw,
        C_raw,
        dd_dt,
        dd_A,
        trap_raw,
        angle_raw,
    ) = jnp.split(
        j_projected_states,
        [s1, s2, s3, s4, s5, s6, s7],
        axis=-1,
    )

    v = v_raw.reshape(num_tokens, num_heads, head_dim)
    if is_mimo:
        B_grouped = jax_rms_norm_per_feature(
            B_raw.reshape(num_tokens, n_groups, r, d_state),
            j_B_norm_weight,
            eps=rms_norm_eps,
        )
        C_grouped = jax_rms_norm_per_feature(
            C_raw.reshape(num_tokens, n_groups, r, d_state),
            j_C_norm_weight,
            eps=rms_norm_eps,
        )
    else:
        B_grouped = jax_rms_norm_per_feature(
            B_raw.reshape(num_tokens, n_groups, d_state),
            j_B_norm_weight,
            eps=rms_norm_eps,
        )
        C_grouped = jax_rms_norm_per_feature(
            C_raw.reshape(num_tokens, n_groups, d_state),
            j_C_norm_weight,
            eps=rms_norm_eps,
        )

    B_normed = broadcast_groups_to_heads(B_grouped, num_heads)
    C_normed = broadcast_groups_to_heads(C_grouped, num_heads)
    q_pre = C_normed + j_C_bias.astype(C_normed.dtype)[None, ...]
    k_pre = B_normed + j_B_bias.astype(B_normed.dtype)[None, ...]
    return z_raw, v, q_pre, k_pre, dd_dt, dd_A, trap_raw, angle_raw


def _discretize_and_apply_rope(
    v: jnp.ndarray,
    q_pre: jnp.ndarray,
    k_pre: jnp.ndarray,
    dd_dt: jnp.ndarray,
    dd_A: jnp.ndarray,
    trap_raw: jnp.ndarray,
    angle_raw: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    j_psi: jnp.ndarray | None,
    prev_angle_seqs: jnp.ndarray,
    effective_query_start_loc: jnp.ndarray,
    active_nonempty_seq: jnp.ndarray,
    has_initial_state: jnp.ndarray,
    *,
    num_rope_angles: int,
    chunk_size: int,
    A_floor: float,
    is_mimo: bool,
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
    """Computes trapezoidal discretization and segmented cumulative 2D RoPE."""
    num_tokens = v.shape[0]
    max_reqs = active_nonempty_seq.shape[0]

    A = jnp.minimum(-jax.nn.softplus(dd_A.astype(jnp.float32)), -A_floor)
    dt = jax.nn.softplus(dd_dt.astype(jnp.float32) + j_dt_bias.astype(jnp.float32))
    adt = A * dt
    trap = jax.nn.sigmoid(trap_raw.astype(jnp.float32))
    gamma = dt * trap
    beta = dt * (1.0 - trap)

    token_indices = jnp.arange(num_tokens)
    seq_id = (
        jnp.searchsorted(effective_query_start_loc, token_indices, side="right") - 1
    )
    seq_id = jnp.clip(seq_id, 0, max_reqs - 1)
    seq_start_for_token = effective_query_start_loc[seq_id]
    seq_end_for_token = effective_query_start_loc[seq_id + 1]

    angle_delta = (
        jnp.tanh(angle_raw.astype(jnp.float32))[:, None, :] * jnp.pi * dt[:, :, None]
    )
    seq_start_tokens = jnp.where(
        active_nonempty_seq, effective_query_start_loc[:max_reqs], num_tokens
    )
    seeded_prev_angles = jnp.where(
        (active_nonempty_seq & has_initial_state)[:, None, None],
        prev_angle_seqs,
        0.0,
    )
    angle_init_offsets = (
        jnp.zeros_like(angle_delta)
        .at[seq_start_tokens]
        .add(seeded_prev_angles, mode="drop")
    )
    angle_with_init = angle_delta + angle_init_offsets
    global_angle_cumsum = _chunked_cumsum_axis0(angle_with_init, block_size=chunk_size)
    prev_seq_end_idx = jnp.maximum(seq_start_for_token - 1, 0)
    base_angle = jnp.where(
        (seq_start_for_token > 0)[:, None, None],
        global_angle_cumsum[prev_seq_end_idx],
        0.0,
    )
    angle_cumsum = jnp.mod(global_angle_cumsum - base_angle, 2.0 * jnp.pi)
    cos = jnp.cos(angle_cumsum)
    sin = jnp.sin(angle_cumsum)

    is_last_token_in_seq = token_indices == (seq_end_for_token - 1)
    beta_shifted = jnp.pad(beta[1:], ((0, 1), (0, 0)))
    scale = gamma + jnp.where(is_last_token_in_seq[:, None], 0.0, beta_shifted)

    if is_mimo:
        assert j_psi is not None
        cos_mimo = cos[:, :, None, :]
        sin_mimo = sin[:, :, None, :]
        q_rot = _rotate_split_half_mimo(q_pre, cos_mimo, sin_mimo, num_rope_angles)
        k_rot = _rotate_split_half_mimo(k_pre, cos_mimo, sin_mimo, num_rope_angles)
        k_scaled = (k_rot.astype(jnp.float32) * scale[:, :, None, None]).astype(v.dtype)
        X_scan = (
            v[:, :, None, :].astype(jnp.float32) * j_psi[None, ...].astype(jnp.float32)
        ).astype(v.dtype)
    else:
        q_rot = _rotate_pairs_siso(q_pre, cos, sin, num_rope_angles)
        k_rot = _rotate_pairs_siso(k_pre, cos, sin, num_rope_angles)
        k_scaled = (k_rot.astype(jnp.float32) * scale[:, :, None]).astype(v.dtype)
        X_scan = v

    return X_scan, q_rot, k_rot, k_scaled, adt, gamma, beta, angle_cumsum


def _augment_initial_states_trapezoidal(
    recurrent_state: jnp.ndarray,
    state_indices: jnp.ndarray,
    prev_x_seqs: jnp.ndarray,
    prev_k_seqs: jnp.ndarray,
    beta: jnp.ndarray,
    safe_start_tokens: jnp.ndarray,
    active_nonempty_seq: jnp.ndarray,
    has_initial_state: jnp.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    d_state: int,
    r: int,
    is_mimo: bool,
) -> jnp.ndarray:
    """Adds cross-step trapezoidal tail beta_{start} * outer(prev_x, prev_k)."""
    max_reqs = active_nonempty_seq.shape[0]
    init_states_for_seqs = recurrent_state[state_indices].astype(jnp.float32)
    beta_at_seq_start = beta[safe_start_tokens]

    if is_mimo:
        prev_x_r = prev_x_seqs.reshape(max_reqs, num_heads, r, head_dim)
        prev_k_r = prev_k_seqs.reshape(max_reqs, num_heads, r, d_state)
        trap_tail = (
            jnp.einsum(
                "shrp,shrn->shpn",
                prev_x_r,
                prev_k_r,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            * beta_at_seq_start[:, :, None, None]
        )
    else:
        trap_tail = (
            prev_x_seqs[:, :, :, None]
            * prev_k_seqs[:, :, None, :]
            * beta_at_seq_start[:, :, None, None]
        )

    return jnp.where(
        (active_nonempty_seq & has_initial_state)[:, None, None, None],
        init_states_for_seqs + trap_tail,
        0.0,
    )


def _eval_single_token_decode(
    X_scan: jnp.ndarray,
    k_scaled: jnp.ndarray,
    q_rot: jnp.ndarray,
    adt: jnp.ndarray,
    init_states_augmented: jnp.ndarray,
    safe_start_tokens: jnp.ndarray,
    is_decode_seq: jnp.ndarray,
    *,
    is_mimo: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluates single-token decode sequences in vectorized form."""
    num_tokens = X_scan.shape[0]
    adt_0 = adt[safe_start_tokens]
    decay_0 = jnp.exp(adt_0)
    S_decayed_1tok = init_states_augmented * decay_0[:, :, None, None]
    X_0 = X_scan[safe_start_tokens]
    B_0 = k_scaled[safe_start_tokens]
    C_0 = q_rot[safe_start_tokens]

    if is_mimo:
        Y_scan_1tok = jnp.einsum(
            "shrn,shpn->shrp",
            C_0,
            S_decayed_1tok.astype(C_0.dtype),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        kv_0 = jnp.einsum(
            "shrp,shrn->shpn",
            X_0,
            B_0,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
    else:
        Y_scan_1tok = jnp.einsum(
            "shn,shpn->shp",
            C_0,
            S_decayed_1tok.astype(C_0.dtype),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        kv_0 = (
            X_0.astype(jnp.float32)[:, :, :, None]
            * B_0.astype(jnp.float32)[:, :, None, :]
        )
    new_rec_1tok = S_decayed_1tok + kv_0
    decode_tok_indices = jnp.where(is_decode_seq, safe_start_tokens, num_tokens)
    return Y_scan_1tok, new_rec_1tok, decode_tok_indices


def _run_ragged_ssd3_scan(
    X_scan: jnp.ndarray,
    k_scaled: jnp.ndarray,
    q_rot: jnp.ndarray,
    adt: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    init_states_augmented: jnp.ndarray,
    Y_scan_1tok: jnp.ndarray,
    new_rec_1tok: jnp.ndarray,
    decode_tok_indices: jnp.ndarray,
    is_decode_seq: jnp.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    d_state: int,
    chunk_size: int,
    r: int,
    is_mimo: bool,
    compute_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs chunked SSD-3 scan for prefill sequences and merges 1-token decodes."""
    (
        X_c,
        B_c,
        C_c,
        decay_c,
        reset_mask,
        last_chunk_indices,
        start_chunk_indices,
        padded_indices_valid,
    ) = pack_inputs_mamba3(
        X_scan,
        k_scaled,
        q_rot,
        adt,
        query_start_loc,
        distribution,
        chunk_size,
        compute_dtype=compute_dtype,
        skip_single_token_seqs=True,
    )
    num_chunks_total = X_c.shape[0]
    init_h_per_chunk = jnp.zeros(
        (num_chunks_total, num_heads, head_dim, d_state),
        dtype=jnp.float32,
    )
    init_h_per_chunk = init_h_per_chunk.at[start_chunk_indices].set(
        init_states_augmented, mode="drop"
    )
    Y_c, new_rec_chunk = ssd3_minimal_discrete_hybrid(
        X_c,
        decay_c,
        B_c,
        C_c,
        last_chunk_indices,
        init_h_per_chunk,
        reset_mask,
        mimo_rank=r,
    )
    if is_mimo:
        Y_packed = (
            Y_c.reshape(num_chunks_total, num_heads, chunk_size, r, head_dim)
            .transpose(0, 2, 1, 3, 4)
            .reshape(-1, num_heads, r, head_dim)
        )
        Y_scan_chunk = Y_packed[padded_indices_valid].astype(jnp.float32)
    else:
        Y_flat = Y_c.transpose(0, 2, 1, 3).reshape(-1, num_heads, head_dim)
        Y_scan_chunk = Y_flat[padded_indices_valid].astype(jnp.float32)

    Y_scan_merged = Y_scan_chunk.at[decode_tok_indices].set(Y_scan_1tok, mode="drop")
    new_rec_merged = jnp.where(
        is_decode_seq[:, None, None, None], new_rec_1tok, new_rec_chunk
    )
    return Y_scan_merged, new_rec_merged


def _apply_mamba3_diag_skip_and_gate(
    Y_scan: jnp.ndarray,
    v: jnp.ndarray,
    X_scan: jnp.ndarray,
    q_pre: jnp.ndarray,
    k_pre: jnp.ndarray,
    gamma: jnp.ndarray,
    z_raw: jnp.ndarray,
    j_D: jnp.ndarray,
    j_psi: jnp.ndarray | None,
    j_phi: jnp.ndarray | None,
    j_zeta: jnp.ndarray | None,
    *,
    num_heads: int,
    head_dim: int,
    is_mimo: bool,
    output_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Applies analytical trapezoidal diagonal, D skip, and MIMO rank-gating."""
    num_tokens = v.shape[0]
    local_intermediate = num_heads * head_dim

    if is_mimo:
        assert j_psi is not None and j_phi is not None and j_zeta is not None
        qk_pre = jnp.einsum(
            "thrn,thsn->thrs",
            q_pre,
            k_pre,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        psi_contract = jnp.einsum(
            "thrs,hsp->thrp",
            qk_pre.astype(q_pre.dtype),
            j_psi.astype(q_pre.dtype),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        diag = (
            gamma[:, :, None, None]
            * v[:, :, None, :].astype(jnp.float32)
            * psi_contract
        )
        d_skip = j_D.astype(jnp.float32)[None, :, None, None] * X_scan.astype(
            jnp.float32
        )
        Y_full = Y_scan + diag + d_skip

        z_reshaped = z_raw.reshape(num_tokens, num_heads, 1, head_dim).astype(
            jnp.float32
        )
        gated = Y_full * jax.nn.silu(z_reshaped * j_zeta[None, ...].astype(jnp.float32))
        Y_combined = jnp.sum(
            gated * j_phi[None, ...].astype(jnp.float32), axis=2
        ).reshape(num_tokens, local_intermediate)
        return Y_combined.astype(output_dtype), None

    qk_dot_gamma = (
        jnp.sum(q_pre.astype(jnp.float32) * k_pre.astype(jnp.float32), axis=-1) * gamma
    )
    diag_plus_d = (j_D.astype(jnp.float32)[None, :] + qk_dot_gamma)[
        :, :, None
    ] * v.astype(jnp.float32)
    Y_final = (
        (Y_scan + diag_plus_d)
        .reshape(num_tokens, local_intermediate)
        .astype(output_dtype)
    )
    return Y_final, z_raw


def _update_mamba3_states(
    aux_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    new_recurrent_states: jnp.ndarray,
    X_scan: jnp.ndarray,
    k_rot: jnp.ndarray,
    angle_cumsum: jnp.ndarray,
    state_indices: jnp.ndarray,
    effective_query_start_loc: jnp.ndarray,
    active_nonempty_seq: jnp.ndarray,
    *,
    num_heads: int,
    dim_x_aux: int,
    dim_k_aux: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Updates recurrent and boundary auxiliary caches for active sequences."""
    num_tokens = X_scan.shape[0]
    max_reqs = active_nonempty_seq.shape[0]

    scatter_state_indices = jnp.where(
        active_nonempty_seq, state_indices, recurrent_state.shape[0]
    )
    updated_recurrent_state = recurrent_state.at[scatter_state_indices].set(
        new_recurrent_states.astype(recurrent_state.dtype), mode="drop"
    )

    last_token_indices = jnp.clip(
        effective_query_start_loc[1 : max_reqs + 1] - 1,
        0,
        jnp.maximum(num_tokens - 1, 0),
    )
    new_prev_x = (
        X_scan[last_token_indices]
        .reshape(max_reqs, num_heads, dim_x_aux)
        .astype(aux_state.dtype)
    )
    new_prev_k = (
        k_rot[last_token_indices]
        .reshape(max_reqs, num_heads, dim_k_aux)
        .astype(aux_state.dtype)
    )
    new_prev_angle = angle_cumsum[last_token_indices].astype(aux_state.dtype)
    new_aux_for_seqs = jnp.concatenate(
        [new_prev_x, new_prev_k, new_prev_angle], axis=-1
    )
    updated_aux_state = aux_state.at[scatter_state_indices].set(
        new_aux_for_seqs, mode="drop"
    )
    return updated_aux_state, updated_recurrent_state


def run_jax_mamba_mixer3_local(
    j_projected_states: jnp.ndarray,
    aux_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_B_norm_weight: jnp.ndarray,
    j_C_norm_weight: jnp.ndarray,
    j_B_bias: jnp.ndarray,
    j_C_bias: jnp.ndarray,
    j_D: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    j_norm_weight: jnp.ndarray | None,
    j_psi: jnp.ndarray | None,
    j_phi: jnp.ndarray | None,
    j_zeta: jnp.ndarray | None,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    num_rope_angles: int,
    chunk_size: int,
    is_mimo: bool = False,
    mimo_rank: int = 1,
    A_floor: float = 1e-4,
    rms_norm_eps: float = 1e-5,
    apply_rms_norm: bool = True,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the local device execution of Mamba-3 Mixer."""
    r = mimo_rank if is_mimo else 1
    num_tokens = j_projected_states.shape[0]
    max_reqs = seq_lens.shape[0]
    num_valid_seqs = distribution[2]

    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(
        valid_loc_mask, query_start_loc, last_valid_loc
    )
    query_lens = (
        effective_query_start_loc[1 : max_reqs + 1]
        - effective_query_start_loc[:max_reqs]
    )
    has_initial_state = (seq_lens - query_lens) > 0
    active_nonempty_seq = (jnp.arange(max_reqs) < num_valid_seqs) & (query_lens > 0)
    is_decode_seq = active_nonempty_seq & (query_lens == 1)
    is_prefill_seq = active_nonempty_seq & (query_lens > 1)
    is_pure_decode = ~jnp.any(is_prefill_seq)

    # 1. Split projected states and compute RMS-normalized Q/K pre-rotations
    z_raw, v, q_pre, k_pre, dd_dt, dd_A, trap_raw, angle_raw = _project_and_norm_qkv(
        j_projected_states,
        j_B_norm_weight,
        j_C_norm_weight,
        j_B_bias,
        j_C_bias,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        r=r,
        is_mimo=is_mimo,
        rms_norm_eps=rms_norm_eps,
    )

    # 2. Unpack per-sequence boundary auxiliary state
    dim_x_aux = r * head_dim
    dim_k_aux = r * d_state
    aux_for_seqs = aux_state[state_indices].astype(jnp.float32)
    prev_x_seqs = aux_for_seqs[..., :dim_x_aux]
    prev_k_seqs = aux_for_seqs[..., dim_x_aux : dim_x_aux + dim_k_aux]
    prev_angle_seqs = aux_for_seqs[..., dim_x_aux + dim_k_aux :]

    # 3. Discretization, cumulative 2D RoPE, and trapezoidal key scaling
    X_scan, q_rot, k_rot, k_scaled, adt, gamma, beta, angle_cumsum = (
        _discretize_and_apply_rope(
            v,
            q_pre,
            k_pre,
            dd_dt,
            dd_A,
            trap_raw,
            angle_raw,
            j_dt_bias,
            j_psi,
            prev_angle_seqs,
            effective_query_start_loc,
            active_nonempty_seq,
            has_initial_state,
            num_rope_angles=num_rope_angles,
            chunk_size=chunk_size,
            A_floor=A_floor,
            is_mimo=is_mimo,
        )
    )

    # 4. Augment initial states with cross-step trapezoidal tail
    safe_start_tokens = jnp.clip(
        effective_query_start_loc[:max_reqs], 0, jnp.maximum(num_tokens - 1, 0)
    )
    init_states_augmented = _augment_initial_states_trapezoidal(
        recurrent_state,
        state_indices,
        prev_x_seqs,
        prev_k_seqs,
        beta,
        safe_start_tokens,
        active_nonempty_seq,
        has_initial_state,
        num_heads=num_heads,
        head_dim=head_dim,
        d_state=d_state,
        r=r,
        is_mimo=is_mimo,
    )

    # 5. Evaluate single-token decode fast path and dispatch ragged scan
    Y_scan_1tok, new_rec_1tok, decode_tok_indices = _eval_single_token_decode(
        X_scan,
        k_scaled,
        q_rot,
        adt,
        init_states_augmented,
        safe_start_tokens,
        is_decode_seq,
        is_mimo=is_mimo,
    )

    def _pure_decode_branch(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
        y_zeros = jnp.zeros(X_scan.shape, dtype=jnp.float32)
        y_scan_out = y_zeros.at[decode_tok_indices].set(Y_scan_1tok, mode="drop")
        return y_scan_out, new_rec_1tok

    def _general_ragged_branch(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _run_ragged_ssd3_scan(
            X_scan,
            k_scaled,
            q_rot,
            adt,
            query_start_loc,
            distribution,
            init_states_augmented,
            Y_scan_1tok,
            new_rec_1tok,
            decode_tok_indices,
            is_decode_seq,
            num_heads=num_heads,
            head_dim=head_dim,
            d_state=d_state,
            chunk_size=chunk_size,
            r=r,
            is_mimo=is_mimo,
            compute_dtype=v.dtype,
        )

    if num_tokens <= 1:
        Y_scan, new_recurrent_states = _pure_decode_branch(None)
    else:
        Y_scan, new_recurrent_states = jax.lax.cond(
            is_pure_decode, _pure_decode_branch, _general_ragged_branch, operand=None
        )

    # 6. Apply diagonal trapezoidal correction, D skip, and MIMO rank-gating
    Y_final, gate_for_norm = _apply_mamba3_diag_skip_and_gate(
        Y_scan,
        v,
        X_scan,
        q_pre,
        k_pre,
        gamma,
        z_raw,
        j_D,
        j_psi,
        j_phi,
        j_zeta,
        num_heads=num_heads,
        head_dim=head_dim,
        is_mimo=is_mimo,
        output_dtype=j_projected_states.dtype,
    )

    # 7. Update recurrent and auxiliary state caches for active sequences
    updated_aux_state, updated_recurrent_state = _update_mamba3_states(
        aux_state,
        recurrent_state,
        new_recurrent_states,
        X_scan,
        k_rot,
        angle_cumsum,
        state_indices,
        effective_query_start_loc,
        active_nonempty_seq,
        num_heads=num_heads,
        dim_x_aux=dim_x_aux,
        dim_k_aux=dim_k_aux,
    )

    # 8. Output gating and optional RMSNorm (in Python when TP all-reduce needed)
    if apply_rms_norm:
        Y_out = jax_gated_rms_norm_mamba3(
            x=Y_final,
            gate=gate_for_norm,
            weight=j_norm_weight,
            eps=rms_norm_eps,
            n_groups=n_groups,
        )
    else:
        Y_out = Y_final.astype(j_projected_states.dtype)

    return (updated_aux_state, updated_recurrent_state), Y_out


def run_jax_mamba_mixer3(
    j_projected_states: jnp.ndarray,
    aux_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_B_norm_weight: jnp.ndarray,
    j_C_norm_weight: jnp.ndarray,
    j_B_bias: jnp.ndarray,
    j_C_bias: jnp.ndarray,
    j_D: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    j_norm_weight: jnp.ndarray | None,
    j_psi: jnp.ndarray | None,
    j_phi: jnp.ndarray | None,
    j_zeta: jnp.ndarray | None,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    *,
    mesh: jax.sharding.Mesh,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    num_rope_angles: int,
    chunk_size: int = 64,
    is_mimo: bool = False,
    mimo_rank: int = 1,
    A_floor: float = 1e-4,
    rms_norm_eps: float = 1e-5,
    apply_rms_norm: bool = True,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the sharded JAX Mamba-3 Mixer over the given TPU mesh."""
    data_axis = "attn_dp" if "attn_dp" in mesh.axis_names else None
    head_axis = _HEAD_SHARDING_AXIS if _HEAD_SHARDING_AXIS in mesh.axis_names else None

    projected_states_spec = P(data_axis, head_axis)
    bias_spec = P(head_axis, None, None) if is_mimo else P(head_axis, None)
    mimo_param_spec = P(head_axis, None, None)
    norm_weight_spec = P(head_axis) if j_norm_weight is not None else None

    in_specs = (
        projected_states_spec,
        P(data_axis, head_axis, None),  # aux_state
        P(data_axis, head_axis, None, None),  # recurrent_state
        P(None),  # j_B_norm_weight
        P(None),  # j_C_norm_weight
        bias_spec,  # j_B_bias
        bias_spec,  # j_C_bias
        P(head_axis),  # j_D
        P(head_axis),  # j_dt_bias
        norm_weight_spec,  # j_norm_weight
        mimo_param_spec if j_psi is not None else None,
        mimo_param_spec if j_phi is not None else None,
        mimo_param_spec if j_zeta is not None else None,
        P(data_axis),  # state_indices
        P(data_axis),  # query_start_loc
        P(),  # distribution
        P(data_axis),  # seq_lens
    )

    out_specs = (
        (
            P(data_axis, head_axis, None),  # updated_aux_state
            P(data_axis, head_axis, None, None),  # updated_recurrent_state
        ),
        P(data_axis, head_axis),  # Y_out
    )

    local_fn = functools.partial(
        run_jax_mamba_mixer3_local,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        num_rope_angles=num_rope_angles,
        chunk_size=chunk_size,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
        A_floor=A_floor,
        rms_norm_eps=rms_norm_eps,
        apply_rms_norm=apply_rms_norm,
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
        aux_state,
        recurrent_state,
        j_B_norm_weight,
        j_C_norm_weight,
        j_B_bias,
        j_C_bias,
        j_D,
        j_dt_bias,
        j_norm_weight,
        j_psi,
        j_phi,
        j_zeta,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
    )
