# Copyright (c) 2026, Dao AI Lab, Goombalab.
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
"""JAX/Pallas implementation of the Mamba-3 Mixer core for TPU.

Adapted from the Mamba-3 selective state space layer in ``state-spaces/mamba``
(Apache-2.0; heavy-tail ``A`` discretization, exponential-trapezoidal carry,
data-dependent continuous 2D RoPE on ``B``/``C``, and MIMO channel-rank
projections ``mimo_x``/``mimo_o``/``mimo_z``) with TPU serving adaptations:
  - Ragged batching over packed token streams using ``prepare_mamba_batch_metadata``
    and linear-pass ``token_sequence_ids``.
  - Per-sequence boundary auxiliary cache ``aux_state`` storing ``(prev_x, prev_k,
    prev_angle)`` gathered and scattered strictly over active non-empty requests.
  - Shared Pallas/JAX ``ssd_minimal_discrete_hybrid(..., strict_causal=True,
    mimo_rank=r)`` for multi-token prefill sequences combined with a compacted
    1-token decode path.
  - Replicated single-device ``shard_map`` execution per TPU worker process
    (multiprocess tensor parallelism runs in PyTorch).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

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
    token_sequence_ids,
)
from vllm_torchtpu.layers.core.ssd_scan import ssd_minimal_discrete_hybrid


def jax_heavy_tail_activation(x: jax.Array) -> jax.Array:
    """Upstream ``mamba_ssm.modules.mamba3.heavy_tail_activation`` in float32."""
    x_f32 = x.astype(jnp.float32)
    neg = jnp.minimum(x_f32, 0.0)
    pos = jnp.maximum(x_f32, 0.0)
    return pos + jnp.reciprocal(1.0 - neg)


def jax_rms_norm_per_feature(
    x: jax.Array,
    weight: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Apply RMSNorm over the trailing ``d_state`` axis for ``B`` and ``C``."""
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    return (x_f32 * jax.lax.rsqrt(variance + eps) * weight.astype(jnp.float32)).astype(
        x.dtype
    )


def _rotate_pairs_siso(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    num_rope_angles: int,
) -> jax.Array:
    """Apply 2D rotary embedding to interleaved pairs ``(2i, 2i+1)`` for SISO."""
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
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    num_rope_angles: int,
) -> jax.Array:
    """Apply 2D split-half rotary embedding across rank channels for MIMO."""
    half = x.shape[-1] // 2
    x0 = x[..., :num_rope_angles].astype(jnp.float32)
    x_mid = x[..., num_rope_angles:half]
    x1 = x[..., half : half + num_rope_angles].astype(jnp.float32)
    x_tail = x[..., half + num_rope_angles :]
    xo0 = (x0 * cos - x1 * sin).astype(x.dtype)
    xo1 = (x0 * sin + x1 * cos).astype(x.dtype)
    return jnp.concatenate([xo0, x_mid, xo1, x_tail], axis=-1)


def _project_and_norm_qkv(
    j_projected_states: jax.Array,
    j_B_norm_weight: jax.Array,
    j_C_norm_weight: jax.Array,
    j_B_bias: jax.Array,
    j_C_bias: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    r: int,
    is_mimo: bool,
    rms_norm_eps: float,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Split projected states and compute RMS-normalized Q/K pre-rotations."""
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
    # Upstream Mamba3 packs B and C as `(b, l, r * g * n)` (`r` outer, `g` middle).
    if is_mimo:
        B_unpacked = B_raw.reshape(num_tokens, r, n_groups, d_state).transpose(
            0, 2, 1, 3
        )
        C_unpacked = C_raw.reshape(num_tokens, r, n_groups, d_state).transpose(
            0, 2, 1, 3
        )
    else:
        B_unpacked = B_raw.reshape(num_tokens, n_groups, d_state)
        C_unpacked = C_raw.reshape(num_tokens, n_groups, d_state)

    B_grouped = jax_rms_norm_per_feature(B_unpacked, j_B_norm_weight, eps=rms_norm_eps)
    C_grouped = jax_rms_norm_per_feature(C_unpacked, j_C_norm_weight, eps=rms_norm_eps)

    B_normed = broadcast_groups_to_heads(B_grouped, num_heads)
    C_normed = broadcast_groups_to_heads(C_grouped, num_heads)
    q_pre = C_normed + j_C_bias.astype(C_normed.dtype)[None, ...]
    k_pre = B_normed + j_B_bias.astype(B_normed.dtype)[None, ...]
    return z_raw, v, q_pre, k_pre, dd_dt, dd_A, trap_raw, angle_raw


def _discretize_and_apply_rope(
    v: jax.Array,
    q_pre: jax.Array,
    k_pre: jax.Array,
    dd_dt: jax.Array,
    dd_A: jax.Array,
    trap_raw: jax.Array,
    angle_raw: jax.Array,
    j_dt_bias: jax.Array,
    j_mimo_x: jax.Array | None,
    prev_angle_seqs: jax.Array,
    effective_query_start_loc: jax.Array,
    seq_start_tokens: jax.Array,
    valid_init_seq: jax.Array,
    *,
    max_reqs: int,
    num_rope_angles: int,
    A_floor: float,
    is_mimo: bool,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Compute trapezoidal discretization and segmented cumulative 2D RoPE."""
    num_tokens = v.shape[0]

    A = jnp.minimum(-jax_heavy_tail_activation(dd_A), -A_floor)
    dt = jax.nn.softplus(dd_dt.astype(jnp.float32) + j_dt_bias.astype(jnp.float32))
    adt = A * dt
    trap = jax.nn.sigmoid(trap_raw.astype(jnp.float32))
    gamma = dt * trap
    beta = dt * (1.0 - trap)

    token_indices = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_id = token_sequence_ids(effective_query_start_loc, num_tokens, max_reqs)
    seq_start_for_token = effective_query_start_loc[seq_id]
    seq_end_for_token = effective_query_start_loc[seq_id + 1]

    angle_delta = (
        jnp.tanh(angle_raw.astype(jnp.float32))[:, None, :] * jnp.pi * dt[:, :, None]
    )
    seeded_prev_angles = jnp.where(valid_init_seq[:, None, None], prev_angle_seqs, 0.0)
    angle_init_offsets = (
        jnp.zeros_like(angle_delta)
        .at[seq_start_tokens]
        .add(seeded_prev_angles, mode="drop")
    )
    step_angles = jnp.mod(angle_delta + angle_init_offsets, 2.0 * jnp.pi)
    is_seq_start = token_indices == seq_start_for_token

    def _seg_angle_combine(
        left: tuple[jax.Array, jax.Array],
        right: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        a_l, r_l = left
        a_r, r_r = right
        combined_angle = jnp.where(
            r_r[:, None, None], a_r, jnp.mod(a_l + a_r, 2.0 * jnp.pi)
        )
        return combined_angle, r_l | r_r

    angle_cumsum, _ = jax.lax.associative_scan(
        _seg_angle_combine, (step_angles, is_seq_start), axis=0
    )
    cos = jnp.cos(angle_cumsum)
    sin = jnp.sin(angle_cumsum)

    is_last_token_in_seq = token_indices == (seq_end_for_token - 1)
    beta_shifted = jnp.pad(beta[1:], ((0, 1), (0, 0)))
    scale = gamma + jnp.where(is_last_token_in_seq[:, None], 0.0, beta_shifted)

    if is_mimo:
        if j_mimo_x is None:
            raise ValueError("j_mimo_x is required when is_mimo=True.")
        cos_mimo = cos[:, :, None, :]
        sin_mimo = sin[:, :, None, :]
        q_rot = _rotate_split_half_mimo(q_pre, cos_mimo, sin_mimo, num_rope_angles)
        k_rot = _rotate_split_half_mimo(k_pre, cos_mimo, sin_mimo, num_rope_angles)
        k_scaled = (k_rot.astype(jnp.float32) * scale[:, :, None, None]).astype(v.dtype)
        X_scan = (
            v[:, :, None, :].astype(jnp.float32)
            * j_mimo_x[None, ...].astype(jnp.float32)
        ).astype(v.dtype)
    else:
        q_rot = _rotate_pairs_siso(q_pre, cos, sin, num_rope_angles)
        k_rot = _rotate_pairs_siso(k_pre, cos, sin, num_rope_angles)
        k_scaled = (k_rot.astype(jnp.float32) * scale[:, :, None]).astype(v.dtype)
        X_scan = v

    return X_scan, q_rot, k_rot, k_scaled, adt, gamma, beta, angle_cumsum


def _augment_initial_states_trapezoidal(
    init_states_compact: jax.Array,
    prev_x_compact: jax.Array,
    prev_k_compact: jax.Array,
    beta: jax.Array,
    safe_start_tokens: jax.Array,
    valid_init_compact: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    d_state: int,
    r: int,
    is_mimo: bool,
) -> jax.Array:
    """Add cross-step trapezoidal tail ``beta_start * outer(prev_x, prev_k)``."""
    k_cap = valid_init_compact.shape[0]
    beta_at_seq_start = beta[safe_start_tokens]

    if is_mimo:
        prev_x_r = prev_x_compact.reshape(k_cap, num_heads, r, head_dim)
        prev_k_r = prev_k_compact.reshape(k_cap, num_heads, r, d_state)
        # Preserve float32 accuracy for cross-step trapezoidal state carry.
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
            prev_x_compact[:, :, :, None]
            * prev_k_compact[:, :, None, :]
            * beta_at_seq_start[:, :, None, None]
        )

    return jnp.where(
        valid_init_compact[:, None, None, None],
        init_states_compact + trap_tail,
        0.0,
    )


def _eval_single_token_decode(
    X_scan: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
    adt: jax.Array,
    init_states_augmented: jax.Array,
    safe_start_tokens: jax.Array,
    compact_is_decode: jax.Array,
    *,
    is_mimo: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate single-token sequences over the ``k_cap`` compacted rows."""
    num_tokens = X_scan.shape[0]
    adt_0 = adt[safe_start_tokens]
    decay_0 = jnp.exp(adt_0)
    S_decayed_1tok = init_states_augmented * decay_0[:, :, None, None]
    X_0 = X_scan[safe_start_tokens]
    B_0 = k_scaled[safe_start_tokens]
    C_0 = q_rot[safe_start_tokens]

    # Preserve float32 state carry and readout accuracy on decode contractions.
    if is_mimo:
        Y_scan_1tok = jnp.einsum(
            "shrn,shpn->shrp",
            C_0.astype(jnp.float32),
            S_decayed_1tok,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        kv_0 = jnp.einsum(
            "shrp,shrn->shpn",
            X_0.astype(jnp.float32),
            B_0.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
    else:
        Y_scan_1tok = jnp.einsum(
            "shn,shpn->shp",
            C_0.astype(jnp.float32),
            S_decayed_1tok,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        kv_0 = (
            X_0.astype(jnp.float32)[:, :, :, None]
            * B_0.astype(jnp.float32)[:, :, None, :]
        )
    new_rec_1tok = S_decayed_1tok + kv_0
    decode_tok_indices = jnp.where(compact_is_decode, safe_start_tokens, num_tokens)
    return Y_scan_1tok, new_rec_1tok, decode_tok_indices


def _run_ragged_ssd3_scan(
    X_scan: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
    adt: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    init_states_augmented: jax.Array,
    valid_init_compact: jax.Array,
    compact_ids: jax.Array,
    compact_valid: jax.Array,
    compact_is_decode: jax.Array,
    num_nonempty_seqs: jax.Array,
    Y_scan_1tok: jax.Array,
    new_rec_1tok: jax.Array,
    decode_tok_indices: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    r: int,
    is_mimo: bool,
    compute_dtype: jax.typing.DTypeLike,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """Run hybrid Pallas/JAX SSD scan on prefill sequences and merge decodes."""
    num_tokens = X_scan.shape[0]
    max_reqs = query_start_loc.shape[0] - 1
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
    active_chunk_compact = compact_valid & active_chunk_seq[compact_ids]
    init_h_per_seq = pack_active_chunk_initial_states(
        init_states_augmented,
        active_chunk_compact,
        jnp.any(valid_init_compact & active_chunk_compact),
        num_active=num_nonempty_seqs,
        max_chunk_seqs=min(max_reqs, max(1, num_tokens // 2)),
    )
    Y_c, new_rec_chunk = ssd_minimal_discrete_hybrid(
        X_c,
        decay_c,
        B_c,
        C_c,
        last_chunk_indices[compact_ids],
        init_h_per_seq,
        reset_mask,
        strict_causal=True,
        mimo_rank=r,
        interpret=interpret,
    )
    valid_c = padded_indices_valid // chunk_size
    valid_l = padded_indices_valid % chunk_size
    if is_mimo:
        Y_scan_chunk = (
            Y_c.reshape(num_chunks_total, num_heads, chunk_size, r, head_dim)[
                valid_c, :, valid_l
            ]
        ).astype(jnp.float32)
    else:
        Y_scan_chunk = Y_c[valid_c, :, valid_l, :].astype(jnp.float32)

    Y_scan_merged = Y_scan_chunk.at[decode_tok_indices].set(Y_scan_1tok, mode="drop")
    new_rec_merged = jnp.where(
        compact_is_decode[:, None, None, None], new_rec_1tok, new_rec_chunk
    )
    return Y_scan_merged, new_rec_merged


def _apply_mamba3_diag_skip_and_gate(
    Y_scan: jax.Array,
    v: jax.Array,
    X_scan: jax.Array,
    q_pre: jax.Array,
    k_pre: jax.Array,
    gamma: jax.Array,
    z_raw: jax.Array,
    j_D: jax.Array,
    j_mimo_x: jax.Array | None,
    j_mimo_o: jax.Array | None,
    j_mimo_z: jax.Array | None,
    *,
    num_heads: int,
    head_dim: int,
    is_mimo: bool,
    output_dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array | None]:
    """Apply analytical trapezoidal diagonal, ``D`` skip, and MIMO rank-gating."""
    num_tokens = v.shape[0]
    local_intermediate = num_heads * head_dim

    if is_mimo:
        if j_mimo_x is None or j_mimo_o is None or j_mimo_z is None:
            raise ValueError(
                "j_mimo_x, j_mimo_o, and j_mimo_z are required when is_mimo=True."
            )
        # Keep analytical MIMO trapezoidal diagonal contractions in float32.
        qk_pre = jnp.einsum(
            "thrn,thsn->thrs",
            q_pre,
            k_pre,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        psi_contract = jnp.einsum(
            "thrs,hsp->thrp",
            qk_pre,
            j_mimo_x.astype(jnp.float32),
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
        gated = Y_full * jax.nn.silu(
            z_reshaped * j_mimo_z[None, ...].astype(jnp.float32)
        )
        Y_combined = jnp.sum(
            gated * j_mimo_o[None, ...].astype(jnp.float32), axis=2
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
    aux_state: jax.Array,
    recurrent_state: jax.Array,
    new_recurrent_states: jax.Array,
    X_scan: jax.Array,
    k_rot: jax.Array,
    angle_cumsum: jax.Array,
    compact_slots: jax.Array,
    compact_ids: jax.Array,
    effective_query_start_loc: jax.Array,
    compact_valid_slot: jax.Array,
    num_nonempty_seqs: jax.Array,
    *,
    num_heads: int,
    dim_x_aux: int,
    dim_k_aux: int,
) -> tuple[jax.Array, jax.Array]:
    """Scatter updated recurrent and boundary auxiliary states for active rows."""
    num_tokens = X_scan.shape[0]
    k_cap = compact_ids.shape[0]

    updated_recurrent_state = scatter_mamba_states(
        recurrent_state,
        compact_slots,
        compact_valid_slot,
        new_recurrent_states,
        num_active=num_nonempty_seqs,
    )

    last_token_indices = jnp.clip(
        effective_query_start_loc[compact_ids + 1] - 1,
        0,
        jnp.maximum(num_tokens - 1, 0),
    )
    new_prev_x = (
        X_scan[last_token_indices]
        .reshape(k_cap, num_heads, dim_x_aux)
        .astype(aux_state.dtype)
    )
    new_prev_k = (
        k_rot[last_token_indices]
        .reshape(k_cap, num_heads, dim_k_aux)
        .astype(aux_state.dtype)
    )
    new_prev_angle = angle_cumsum[last_token_indices].astype(aux_state.dtype)
    new_aux_for_seqs = jnp.concatenate(
        [new_prev_x, new_prev_k, new_prev_angle], axis=-1
    )
    updated_aux_state = scatter_mamba_states(
        aux_state,
        compact_slots,
        compact_valid_slot,
        new_aux_for_seqs,
        num_active=num_nonempty_seqs,
    )
    return updated_aux_state, updated_recurrent_state


def run_jax_mamba_mixer3_local(
    j_projected_states: jax.Array,
    aux_state: jax.Array,
    recurrent_state: jax.Array,
    j_B_norm_weight: jax.Array,
    j_C_norm_weight: jax.Array,
    j_B_bias: jax.Array,
    j_C_bias: jax.Array,
    j_D: jax.Array,
    j_dt_bias: jax.Array,
    j_norm_weight: jax.Array | None,
    j_mimo_x: jax.Array | None,
    j_mimo_o: jax.Array | None,
    j_mimo_z: jax.Array | None,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
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
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run local per-rank Mamba-3 mixer over a ragged batch."""
    r = mimo_rank if is_mimo else 1
    num_tokens = j_projected_states.shape[0]
    max_reqs = seq_lens.shape[0]
    num_slots = recurrent_state.shape[0]

    meta = prepare_mamba_batch_metadata(
        query_start_loc,
        seq_lens,
        distribution,
        state_indices,
        num_tokens=num_tokens,
        num_slots=num_slots,
    )
    # Unlike Mamba-2 (where `ragged_causal_conv1d` splits decode vs prefill at
    # `distribution[0]`), Mamba-3 evaluates every 1-token sequence in
    # `_eval_single_token_decode` regardless of `distribution[0]`, so the
    # chunked SSD-3 scan can be skipped whenever no active request has `q_len > 1`.
    no_prefill_seqs = ~jnp.any(meta.is_prefill_seq)

    compact_ids = meta.compact_seq_ids
    compact_slots = state_indices[compact_ids]
    compact_safe_slots = meta.safe_state_indices[compact_ids]
    compact_valid_slot = meta.compact_valid & meta.valid_state_slot[compact_ids]
    valid_init_compact = compact_valid_slot & meta.has_initial_state[compact_ids]
    compact_is_decode = meta.compact_valid & meta.is_decode_seq[compact_ids]
    safe_start_tokens = jnp.clip(
        meta.effective_query_start_loc[compact_ids],
        0,
        jnp.maximum(num_tokens - 1, 0),
    )
    seq_start_tokens = jnp.where(meta.compact_valid, safe_start_tokens, num_tokens)

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

    dim_x_aux = r * head_dim
    dim_k_aux = r * d_state
    aux_for_seqs = gather_active_mamba_states(
        aux_state,
        compact_safe_slots,
        valid_init_compact,
        num_active=meta.num_nonempty_seqs,
    )
    prev_x_seqs = aux_for_seqs[..., :dim_x_aux]
    prev_k_seqs = aux_for_seqs[..., dim_x_aux : dim_x_aux + dim_k_aux]
    prev_angle_seqs = aux_for_seqs[..., dim_x_aux + dim_k_aux :]

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
            j_mimo_x,
            prev_angle_seqs,
            meta.effective_query_start_loc,
            seq_start_tokens,
            valid_init_compact,
            max_reqs=max_reqs,
            num_rope_angles=num_rope_angles,
            A_floor=A_floor,
            is_mimo=is_mimo,
        )
    )

    init_states_compact = gather_active_mamba_states(
        recurrent_state,
        compact_safe_slots,
        valid_init_compact,
        num_active=meta.num_nonempty_seqs,
    )
    init_states_augmented = _augment_initial_states_trapezoidal(
        init_states_compact,
        prev_x_seqs,
        prev_k_seqs,
        beta,
        safe_start_tokens,
        valid_init_compact,
        num_heads=num_heads,
        head_dim=head_dim,
        d_state=d_state,
        r=r,
        is_mimo=is_mimo,
    )

    Y_scan_1tok, new_rec_1tok, decode_tok_indices = _eval_single_token_decode(
        X_scan,
        k_scaled,
        q_rot,
        adt,
        init_states_augmented,
        safe_start_tokens,
        compact_is_decode,
        is_mimo=is_mimo,
    )

    def _pure_decode_branch(_: None) -> tuple[jax.Array, jax.Array]:
        y_zeros = jnp.zeros(X_scan.shape, dtype=jnp.float32)
        y_scan_out = y_zeros.at[decode_tok_indices].set(Y_scan_1tok, mode="drop")
        return y_scan_out, new_rec_1tok

    def _general_ragged_branch(_: None) -> tuple[jax.Array, jax.Array]:
        return _run_ragged_ssd3_scan(
            X_scan,
            k_scaled,
            q_rot,
            adt,
            query_start_loc,
            distribution,
            init_states_augmented,
            valid_init_compact,
            compact_ids,
            meta.compact_valid,
            compact_is_decode,
            meta.num_nonempty_seqs,
            Y_scan_1tok,
            new_rec_1tok,
            decode_tok_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            chunk_size=chunk_size,
            r=r,
            is_mimo=is_mimo,
            compute_dtype=v.dtype,
            interpret=interpret,
        )

    if num_tokens <= 1:
        Y_scan, new_recurrent_states = _pure_decode_branch(None)
    else:
        Y_scan, new_recurrent_states = jax.lax.cond(
            no_prefill_seqs, _pure_decode_branch, _general_ragged_branch, operand=None
        )

    Y_final, gate_for_norm = _apply_mamba3_diag_skip_and_gate(
        Y_scan,
        v,
        X_scan,
        q_pre,
        k_pre,
        gamma,
        z_raw,
        j_D,
        j_mimo_x,
        j_mimo_o,
        j_mimo_z,
        num_heads=num_heads,
        head_dim=head_dim,
        is_mimo=is_mimo,
        output_dtype=j_projected_states.dtype,
    )

    updated_aux_state, updated_recurrent_state = _update_mamba3_states(
        aux_state,
        recurrent_state,
        new_recurrent_states,
        X_scan,
        k_rot,
        angle_cumsum,
        compact_slots,
        compact_ids,
        meta.effective_query_start_loc,
        compact_valid_slot,
        meta.num_nonempty_seqs,
        num_heads=num_heads,
        dim_x_aux=dim_x_aux,
        dim_k_aux=dim_k_aux,
    )

    if apply_rms_norm:
        Y_out = jax_gated_rms_norm(
            x=Y_final,
            gate=gate_for_norm,
            weight=j_norm_weight,
            eps=rms_norm_eps,
            n_groups=n_groups,
        )
    else:
        Y_out = Y_final.astype(j_projected_states.dtype)

    return updated_aux_state, updated_recurrent_state, Y_out


def run_jax_mamba_mixer3(
    j_projected_states: jax.Array,
    aux_state: jax.Array,
    recurrent_state: jax.Array,
    j_B_norm_weight: jax.Array,
    j_C_norm_weight: jax.Array,
    j_B_bias: jax.Array,
    j_C_bias: jax.Array,
    j_D: jax.Array,
    j_dt_bias: jax.Array,
    j_norm_weight: jax.Array | None,
    j_mimo_x: jax.Array | None,
    j_mimo_o: jax.Array | None,
    j_mimo_z: jax.Array | None,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    num_rope_angles: int,
    chunk_size: int = TPU_SSD_BLOCK_LEN,
    is_mimo: bool = False,
    mimo_rank: int = 1,
    A_floor: float = 1e-4,
    rms_norm_eps: float = 1e-5,
    apply_rms_norm: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run Mamba-3 mixer over the worker's single-device TPU mesh."""
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
        interpret=interpret,
    )
    return run_replicated_shard_map(
        local_fn,
        mesh,
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
        j_mimo_x,
        j_mimo_o,
        j_mimo_z,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
    )
