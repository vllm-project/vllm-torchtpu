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
(Apache-2.0; exponential-trapezoidal discretization, data-dependent continuous
2D RoPE on ``B``/``C``, and MIMO channel-rank projections ``Psi``/``Phi``/``Zeta``)
with the following TPU serving adaptations:
  - Ragged batching over packed token streams using ``prepare_mamba_batch_metadata``
    and linear-pass ``token_sequence_ids``.
  - Per-sequence boundary auxiliary cache ``aux_state`` storing ``(prev_x, prev_k,
    prev_angle)`` so cumulative RoPE phase and the cross-step trapezoidal tail
    ``beta_start * outer(prev_x, prev_k)`` carry seamlessly across chunked
    prefills and 1-token decodes.
  - Shared Pallas/JAX ``ssd_minimal_discrete_hybrid(..., strict_causal=True,
    mimo_rank=r)`` for multi-token prefill sequences combined with a vectorized
    1-token decode path.
  - Replicated single-device ``shard_map`` execution per TPU worker process
    (multiprocess tensor parallelism runs in PyTorch).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.ssd_scan import ssd_minimal_discrete_hybrid
from vllm_torchtpu.layers.core.mamba_utils import (
    broadcast_groups_to_heads,
    jax_gated_rms_norm,
    pack_ragged_ssm_chunks,
    pack_sequence_initial_states,
    prepare_mamba_batch_metadata,
    run_replicated_shard_map,
    scatter_mamba_states,
    token_sequence_ids,
)


def jax_rms_norm_per_feature(
    x: jax.Array,
    weight: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Apply RMSNorm over the trailing ``d_state`` axis for ``B`` and ``C``."""
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = (x_f32 * jax.lax.rsqrt(variance + eps)).astype(x.dtype)
    return x_norm * weight.astype(x.dtype)


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


def _chunked_cumsum_axis0(x: jax.Array, block_size: int = 64) -> jax.Array:
    """Compute ``jnp.cumsum(x, axis=0)`` using two-level blocking."""
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
    bc_shape = (
        (num_tokens, n_groups, r, d_state)
        if is_mimo
        else (num_tokens, n_groups, d_state)
    )
    B_grouped = jax_rms_norm_per_feature(
        B_raw.reshape(bc_shape), j_B_norm_weight, eps=rms_norm_eps
    )
    C_grouped = jax_rms_norm_per_feature(
        C_raw.reshape(bc_shape), j_C_norm_weight, eps=rms_norm_eps
    )

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
    j_psi: jax.Array | None,
    prev_angle_seqs: jax.Array,
    effective_query_start_loc: jax.Array,
    active_nonempty_seq: jax.Array,
    valid_init_seq: jax.Array,
    *,
    num_rope_angles: int,
    chunk_size: int,
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
    max_reqs = active_nonempty_seq.shape[0]

    A = jnp.minimum(-jax.nn.softplus(dd_A.astype(jnp.float32)), -A_floor)
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
    seq_start_tokens = jnp.where(
        active_nonempty_seq, effective_query_start_loc[:max_reqs], num_tokens
    )
    seeded_prev_angles = jnp.where(valid_init_seq[:, None, None], prev_angle_seqs, 0.0)
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
        if j_psi is None:
            raise ValueError("j_psi is required when is_mimo=True.")
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
    recurrent_state: jax.Array,
    safe_state_indices: jax.Array,
    prev_x_seqs: jax.Array,
    prev_k_seqs: jax.Array,
    beta: jax.Array,
    safe_start_tokens: jax.Array,
    valid_init_seq: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    d_state: int,
    r: int,
    is_mimo: bool,
) -> jax.Array:
    """Add cross-step trapezoidal tail ``beta_start * outer(prev_x, prev_k)``."""
    max_reqs = valid_init_seq.shape[0]
    init_states_for_seqs = recurrent_state[safe_state_indices].astype(jnp.float32)
    beta_at_seq_start = beta[safe_start_tokens]

    if is_mimo:
        prev_x_r = prev_x_seqs.reshape(max_reqs, num_heads, r, head_dim)
        prev_k_r = prev_k_seqs.reshape(max_reqs, num_heads, r, d_state)
        # Keep float32 precision on the float32 auxiliary cache outer product so
        # cross-step recurrent carry matches contiguous prefill accumulation.
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
        valid_init_seq[:, None, None, None],
        init_states_for_seqs + trap_tail,
        0.0,
    )


def _eval_single_token_decode(
    X_scan: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
    adt: jax.Array,
    init_states_augmented: jax.Array,
    safe_start_tokens: jax.Array,
    is_decode_seq: jax.Array,
    *,
    is_mimo: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate single-token sequences directly without chunk packing."""
    num_tokens = X_scan.shape[0]
    adt_0 = adt[safe_start_tokens]
    decay_0 = jnp.exp(adt_0)
    S_decayed_1tok = init_states_augmented * decay_0[:, :, None, None]
    X_0 = X_scan[safe_start_tokens]
    B_0 = k_scaled[safe_start_tokens]
    C_0 = q_rot[safe_start_tokens]

    # Keep Precision.HIGHEST on float32 state carry readout and MIMO outer
    # product so 1-token decode preserves float32 state fidelity (matching
    # chunk_state_pallas / chunk_scan_pallas).
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
    decode_tok_indices = jnp.where(is_decode_seq, safe_start_tokens, num_tokens)
    return Y_scan_1tok, new_rec_1tok, decode_tok_indices


def _run_ragged_ssd3_scan(
    X_scan: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
    adt: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    init_states_augmented: jax.Array,
    Y_scan_1tok: jax.Array,
    new_rec_1tok: jax.Array,
    decode_tok_indices: jax.Array,
    is_decode_seq: jax.Array,
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
    init_h_per_seq = pack_sequence_initial_states(
        init_states_augmented, active_chunk_seq
    )
    Y_c, new_rec_chunk = ssd_minimal_discrete_hybrid(
        X_c,
        decay_c,
        B_c,
        C_c,
        last_chunk_indices,
        init_h_per_seq,
        reset_mask,
        strict_causal=True,
        mimo_rank=r,
        interpret=interpret,
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
    Y_scan: jax.Array,
    v: jax.Array,
    X_scan: jax.Array,
    q_pre: jax.Array,
    k_pre: jax.Array,
    gamma: jax.Array,
    z_raw: jax.Array,
    j_D: jax.Array,
    j_psi: jax.Array | None,
    j_phi: jax.Array | None,
    j_zeta: jax.Array | None,
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
        if j_psi is None or j_phi is None or j_zeta is None:
            raise ValueError("j_psi, j_phi, and j_zeta are required when is_mimo=True.")
        # Keep Precision.HIGHEST so float32 inputs in tests and small rank-R
        # contractions match the analytical trapezoidal diagonal without bf16
        # pre-rounding before summing with float32 `Y_scan`.
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
            j_psi.astype(jnp.float32),
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
    aux_state: jax.Array,
    recurrent_state: jax.Array,
    new_recurrent_states: jax.Array,
    X_scan: jax.Array,
    k_rot: jax.Array,
    angle_cumsum: jax.Array,
    state_indices: jax.Array,
    effective_query_start_loc: jax.Array,
    valid_state_slot: jax.Array,
    *,
    num_heads: int,
    dim_x_aux: int,
    dim_k_aux: int,
) -> tuple[jax.Array, jax.Array]:
    """Scatter updated recurrent and boundary auxiliary states into caches."""
    num_tokens = X_scan.shape[0]
    max_reqs = valid_state_slot.shape[0]

    updated_recurrent_state = scatter_mamba_states(
        recurrent_state, state_indices, valid_state_slot, new_recurrent_states
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
    updated_aux_state = scatter_mamba_states(
        aux_state, state_indices, valid_state_slot, new_aux_for_seqs
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
    j_psi: jax.Array | None,
    j_phi: jax.Array | None,
    j_zeta: jax.Array | None,
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
    valid_init_seq = meta.valid_state_slot & meta.has_initial_state
    no_prefill_seqs = ~jnp.any(meta.is_prefill_seq)

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
    aux_for_seqs = aux_state[meta.safe_state_indices].astype(jnp.float32)
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
            j_psi,
            prev_angle_seqs,
            meta.effective_query_start_loc,
            meta.active_nonempty_seq,
            valid_init_seq,
            num_rope_angles=num_rope_angles,
            chunk_size=chunk_size,
            A_floor=A_floor,
            is_mimo=is_mimo,
        )
    )

    safe_start_tokens = jnp.clip(
        meta.effective_query_start_loc[:max_reqs], 0, jnp.maximum(num_tokens - 1, 0)
    )
    init_states_augmented = _augment_initial_states_trapezoidal(
        recurrent_state,
        meta.safe_state_indices,
        prev_x_seqs,
        prev_k_seqs,
        beta,
        safe_start_tokens,
        valid_init_seq,
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
        meta.is_decode_seq,
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
            Y_scan_1tok,
            new_rec_1tok,
            decode_tok_indices,
            meta.is_decode_seq,
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
        j_psi,
        j_phi,
        j_zeta,
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
        state_indices,
        meta.effective_query_start_loc,
        meta.valid_state_slot,
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
    j_psi: jax.Array | None,
    j_phi: jax.Array | None,
    j_zeta: jax.Array | None,
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
    chunk_size: int = 64,
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
        j_psi,
        j_phi,
        j_zeta,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
    )
