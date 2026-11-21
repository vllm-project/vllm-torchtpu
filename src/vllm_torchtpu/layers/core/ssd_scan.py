# Copyright (c) 2024, Albert Gu and Tri Dao.
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
"""Hybrid JAX/Pallas chunked SSD scan orchestration for Mamba-2 and Mamba-3.

Adapted from the four-step chunked Selective State Space Dual (SSD) formulation
in ``state-spaces/mamba`` (``mamba_ssm.modules.ssd_minimal.ssd_minimal_discrete``,
Apache-2.0) and Mamba-3 SSD-3. Dispatches Steps 2 and 4 to the Pallas TPU
kernels in ``vllm_torchtpu.kernels.ssd_scan`` (``chunk_state_pallas`` and
``chunk_scan_pallas``) while running intra-chunk segment sums, ``einsum``, and
the inter-chunk ``jax.lax.scan`` recurrence in native JAX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.ssd_scan import (
    chunk_scan_pallas,
    chunk_state_pallas,
)


def _segsum_jax(
    x: jax.Array,
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
) -> jax.Array:
    """Computes lower-triangular segment sums of ``x`` along its last axis.

    For ``x`` of shape ``(..., T)`` with ``mimo_rank == 1``, returns ``out`` of
    shape ``(..., T, T)`` where:
      - ``out[..., i, j] = sum(x[..., j + 1 : i + 1])`` for ``i > j``
      - ``out[..., i, i] = 0.0`` if ``not strict_causal`` else ``-inf``
      - ``out[..., i, j] = -inf`` for ``i < j``

    When ``mimo_rank > 1`` (``T = n_tok * mimo_rank``), the causal mask is
    applied at the token block level ``(n_tok, n_tok)`` and repeated by
    ``mimo_rank`` along both axes so all ``mimo_rank`` sub-channels of token
    ``s`` are included (or excluded under ``strict_causal=True`` when
    ``s >= l``) together.
    """
    if mimo_rank < 1:
        raise ValueError(f"Expected mimo_rank >= 1, got {mimo_rank}.")
    T = x.shape[-1]
    if T % mimo_rank != 0:
        raise ValueError(
            f"Expected last dim ({T}) divisible by mimo_rank ({mimo_rank})."
        )

    x_expanded = jnp.expand_dims(x, axis=-1)
    x_repeated = jnp.broadcast_to(x_expanded, x.shape + (T,))

    mask_subdiag = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x_masked = jnp.where(mask_subdiag, x_repeated, 0.0)
    x_segsum = jnp.cumsum(x_masked, axis=-2)

    k_diag = -1 if strict_causal else 0
    if mimo_rank > 1:
        n_tok = T // mimo_rank
        tok_mask = jnp.tril(jnp.ones((n_tok, n_tok), dtype=bool), k=k_diag)
        mask_keep = jnp.repeat(
            jnp.repeat(tok_mask, mimo_rank, axis=0), mimo_rank, axis=1
        )
    else:
        mask_keep = (
            mask_subdiag if strict_causal else jnp.tril(jnp.ones((T, T), dtype=bool))
        )

    return jnp.where(mask_keep, x_segsum, -jnp.inf)


def ssd_minimal_discrete_hybrid(
    X_c: jax.Array,
    A_c: jax.Array,
    B_c: jax.Array,
    C_c: jax.Array,
    last_chunk_indices: jax.Array,
    initial_states: jax.Array | None = None,
    reset_mask: jax.Array | None = None,
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Hybrid JAX/Pallas Selective State Space (SSD) scan for Mamba-2 and Mamba-3.

    Implements the discrete SSD scan using Pallas TPU kernels
    (``chunk_state_pallas`` and ``chunk_scan_pallas``) for intra-chunk state
    generation and state-to-output conversion, and native JAX for cumulative
    sums and ``float32`` sequential inter-chunk recurrence.

    Caller invariants for ragged/padded batches:
      - Sub-chunk tail padding: Padded tail positions in a sequence's final
        chunk must set ``A_c = 0.0`` (so ``exp(A_c) == 1.0``, preserving the last
        valid token's state across padded positions) and ``X_c = 0.0`` (or
        ``B_c = 0.0``, preventing padded positions from adding state updates).
      - Finite log-decay ``A_c``: All entries of ``A_c`` must be finite
        (``<= 0.0``, never ``-inf``), because ``-inf - (-inf)`` in
        ``A_cumsum[..., -1:] - A_cumsum`` produces ``NaN``.
      - Checkpoint & final state gathering: ``last_chunk_indices`` of shape
        ``(num_out_states,)`` may index one final chunk per sequence or arbitrary
        intermediate/final chunk indices in ``[0, n_chunks)`` (e.g., for aligned
        prefix caching or speculative decoding). Inactive padding slots may pass
        ``-1`` sentinels, which are clamped to chunk ``0`` via
        ``jnp.maximum(..., 0)`` without clamping upper-bound indices.

    Args:
      X_c: ``(n_chunks, n_heads, block_len, d_head)`` discretized input tensor,
        where ``block_len == n_tok * mimo_rank`` for MIMO models.
      A_c: Finite log decay ``(n_chunks, n_heads, block_len)``; when
        ``mimo_rank > 1``, either per-token
        ``(n_chunks, n_heads, block_len // mimo_rank)`` or rank-expanded
        ``(n_chunks, n_heads, block_len)`` is accepted.
      B_c: ``(n_chunks, n_groups, block_len, d_state)`` input/key projection
        where ``n_groups`` divides ``n_heads``.
      C_c: ``(n_chunks, n_groups, block_len, d_state)`` output/query projection
        where ``n_groups == B_c.shape[1]``.
      last_chunk_indices: ``(num_out_states,)`` chunk indices in
        ``[0, n_chunks)`` (or ``-1`` for inactive padding slots) to gather from
        outgoing chunk states.
      initial_states: Optional ``(num_seqs, n_heads, d_head, d_state)``
        per-sequence initial states indexed in sequence order at chunks where
        ``reset_mask`` is ``True``. When ``None``, sequence starts reset directly
        to zero inside the scan without allocating a 4D zero buffer in HBM.
      reset_mask: Optional ``(n_chunks,)`` boolean sequence-start mask.
      strict_causal: Exclude intra-token ``s == l`` contributions when True
        (used by Mamba-3, which adds the trapezoidal diagonal analytically).
      mimo_rank: MIMO channel rank ``R >= 1`` flattened into ``block_len``.
      interpret: Run Pallas kernels in host interpret mode (CPU tests only).

    Returns:
      ``(Y, final_states)`` where ``Y`` has shape
      ``(n_chunks, n_heads, block_len, d_head)`` in ``X_c.dtype`` and
      ``final_states`` has shape ``(num_out_states, n_heads, d_head, d_state)``
      in ``float32``.
    """
    if X_c.ndim != 4 or min(X_c.shape) <= 0:
        raise ValueError(
            "Expected positive 4D X_c (n_chunks, n_heads, block_len, d_head), "
            f"got {X_c.shape}."
        )
    n_chunks, n_heads, block_len, d_head = X_c.shape
    if mimo_rank < 1:
        raise ValueError(f"Expected mimo_rank >= 1, got {mimo_rank}.")
    if block_len % mimo_rank != 0:
        raise ValueError(
            f"Expected block_len ({block_len}) divisible by mimo_rank ({mimo_rank})."
        )
    if mimo_rank == 1:
        if A_c.shape != (n_chunks, n_heads, block_len):
            raise ValueError(
                f"Expected A_c shape {(n_chunks, n_heads, block_len)}, got {A_c.shape}."
            )
    else:
        n_tok = block_len // mimo_rank
        valid_a_shapes = (
            (n_chunks, n_heads, n_tok),
            (n_chunks, n_heads, block_len),
        )
        if A_c.shape not in valid_a_shapes:
            raise ValueError(
                f"Expected A_c shape in {valid_a_shapes} for mimo_rank={mimo_rank}, "
                f"got {A_c.shape}."
            )

    if (
        B_c.ndim != 4
        or B_c.shape[0] != n_chunks
        or B_c.shape[2] != block_len
        or B_c.shape[1] <= 0
        or B_c.shape[3] <= 0
        or n_heads % B_c.shape[1] != 0
    ):
        raise ValueError(
            "Expected positive 4D B_c (n_chunks, n_groups, block_len, d_state) "
            f"with leading/chunk dims ({n_chunks}, n_groups, {block_len}) and "
            f"n_groups dividing n_heads={n_heads}, got {B_c.shape}."
        )
    n_groups = B_c.shape[1]
    d_state = B_c.shape[3]
    if C_c.shape != (n_chunks, n_groups, block_len, d_state):
        raise ValueError(
            f"Expected C_c shape {(n_chunks, n_groups, block_len, d_state)}, "
            f"got {C_c.shape}."
        )
    if last_chunk_indices.ndim != 1 or last_chunk_indices.shape[0] <= 0:
        raise ValueError(
            "Expected non-empty 1D last_chunk_indices (num_out_states,), "
            f"got {last_chunk_indices.shape}."
        )

    if initial_states is not None and (
        initial_states.ndim != 4
        or initial_states.shape[0] <= 0
        or initial_states.shape[1:] != (n_heads, d_head, d_state)
    ):
        raise ValueError(
            "Expected initial_states shape (num_seqs, "
            f"{n_heads}, {d_head}, {d_state}) with num_seqs > 0, "
            f"got {initial_states.shape}."
        )
    if reset_mask is not None and reset_mask.shape != (n_chunks,):
        raise ValueError(
            f"Expected reset_mask shape {(n_chunks,)}, got {reset_mask.shape}."
        )

    if reset_mask is None:
        reset_mask = jnp.zeros((n_chunks,), dtype=bool).at[0].set(True)

    if mimo_rank > 1:
        n_tok = block_len // mimo_rank
        A_raw_f32 = A_c.astype(jnp.float32)
        if A_raw_f32.shape[-1] == block_len:
            A_tok = jnp.sum(
                A_raw_f32.reshape(n_chunks, n_heads, n_tok, mimo_rank), axis=-1
            )
        else:
            A_tok = A_raw_f32
        # Concentrate each token's decay at sub-channel 0 and 0.0 on 1..R-1 so
        # cumulative decay `A_cumsum` applies uniformly across all R sub-channels
        # of a token before readout and state accumulation in Pallas.
        A_zeros = jnp.zeros(
            (n_chunks, n_heads, n_tok, mimo_rank - 1), dtype=jnp.float32
        )
        A_c_f32 = jnp.concatenate([A_tok[..., None], A_zeros], axis=-1).reshape(
            n_chunks, n_heads, block_len
        )
    else:
        A_c_f32 = A_c.astype(jnp.float32)

    A_cumsum = jnp.cumsum(A_c_f32, axis=-1)

    # 1. Intra-chunk diagonal output (Y_diag). Keep Precision.HIGHEST so XLA TPU
    # does not truncate the float32 segment-decay matrix `L` to bfloat16 before
    # contracting across `block_len` (which degrades float32 Y_diag max abs error
    # from <1e-5 to >3e-3 on 64-token chunks).
    L = jnp.exp(_segsum_jax(A_c_f32, strict_causal=strict_causal, mimo_rank=mimo_rank))
    if n_groups == n_heads:
        Y_diag = jnp.einsum(
            "chln,chsn,chls,chsp->chlp",
            C_c,
            B_c,
            L,
            X_c,
            precision=jax.lax.Precision.HIGHEST,
        )
    else:
        heads_per_group = n_heads // n_groups
        L_grouped = L.reshape(n_chunks, n_groups, heads_per_group, block_len, block_len)
        X_grouped = X_c.reshape(n_chunks, n_groups, heads_per_group, block_len, d_head)
        Y_diag = jnp.einsum(
            "cgln,cgsn,cghls,cghsp->cghlp",
            C_c,
            B_c,
            L_grouped,
            X_grouped,
            precision=jax.lax.Precision.HIGHEST,
        ).reshape(n_chunks, n_heads, block_len, d_head)

    # 2. Intra-chunk state generation in float32 (Pallas). Place the size-1
    # axis in the sublane dimension `(n_chunks, n_heads, 1, block_len)` so TPU
    # VMEM pads `1` to `8` sublanes instead of `128` lanes.
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)[:, :, None, :]
    states_c = chunk_state_pallas(B_c, decay_states, X_c, interpret=interpret)

    # 3. Inter-chunk recurrence in float32, resetting carry at `reset_mask` starts.
    decay_chunk = jnp.exp(A_cumsum[..., -1])
    init_carry = jnp.zeros((n_heads, d_head, d_state), dtype=jnp.float32)

    if initial_states is None:

        def scan_body_zero_init(carry, args):
            states_c_c, decay_chunk_c, reset_c = args
            incoming_state = jnp.where(reset_c, jnp.zeros_like(carry), carry)
            decay_exp = decay_chunk_c[:, None, None]
            next_state = incoming_state * decay_exp + states_c_c
            return next_state, (incoming_state, next_state)

        _, (states_incoming, states_outgoing) = jax.lax.scan(
            scan_body_zero_init,
            init_carry,
            (states_c, decay_chunk, reset_mask),
        )
    else:
        initial_states_f32 = initial_states.astype(jnp.float32)
        num_init_seqs = initial_states_f32.shape[0]

        def scan_body_with_init(carry_and_seq, args):
            carry, seq_idx = carry_and_seq
            states_c_c, decay_chunk_c, reset_c = args
            safe_seq_idx = jnp.minimum(seq_idx, num_init_seqs - 1)
            incoming_state = jnp.where(reset_c, initial_states_f32[safe_seq_idx], carry)
            next_seq_idx = jnp.where(reset_c, seq_idx + jnp.int32(1), seq_idx)
            decay_exp = decay_chunk_c[:, None, None]
            next_state = incoming_state * decay_exp + states_c_c
            return (next_state, next_seq_idx), (incoming_state, next_state)

        _, (states_incoming, states_outgoing) = jax.lax.scan(
            scan_body_with_init,
            (init_carry, jnp.int32(0)),
            (states_c, decay_chunk, reset_mask),
        )

    # 4. State to output conversion (Pallas). Keep Y_off in float32 until it is
    # summed with Y_diag so opposite-sign terms do not suffer pre-cancellation
    # bfloat16 rounding before the single output cast. The decay from the chunk
    # start is applied here, where XLA fuses it into the sum.
    Y_off = chunk_scan_pallas(
        C_c, states_incoming, out_dtype=jnp.float32, interpret=interpret
    )
    state_decay_out = jnp.exp(A_cumsum)[..., None]
    Y = (Y_diag.astype(jnp.float32) + Y_off * state_decay_out).astype(X_c.dtype)

    # Clamp negative `-1` padding sentinels to `0` without clamping the upper
    # bound so out-of-range caller indices are not masked.
    safe_last_chunk_indices = jnp.maximum(last_chunk_indices, 0)
    final_states = states_outgoing[safe_last_chunk_indices]
    return Y, final_states
