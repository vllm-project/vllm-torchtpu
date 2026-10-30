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
"""JAX implementation of Mamba-3 Selective State Space (SSD-3) scan."""

import jax
import jax.numpy as jnp


def _segsum_strict_jax(x: jnp.ndarray, mimo_rank: int = 1) -> jnp.ndarray:
    """Computes strict causal segment sums (s < l) for Mamba-3 intra-chunk decay.

    Unlike Mamba-2 (which includes the diagonal s <= l inside the SSD scan),
    Mamba-3 uses a strictly lower-triangular intra-chunk mask (s < l) because
    the diagonal s == l contribution is governed by the trapezoidal weight
    gamma_t = dt_t * trap_t and added analytically outside the scan.
    """
    T = x.shape[-1]
    x_expanded = jnp.expand_dims(x, axis=-1)
    x_repeated = jnp.broadcast_to(x_expanded, x.shape + (T,))

    mask_pos = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x_masked = jnp.where(mask_pos, x_repeated, 0.0)
    x_segsum = jnp.cumsum(x_masked, axis=-2)

    if mimo_rank > 1:
        n_tok = T // mimo_rank
        tok_mask = jnp.tril(jnp.ones((n_tok, n_tok), dtype=bool), k=-1)
        mask_strict = jnp.repeat(
            jnp.repeat(tok_mask, mimo_rank, axis=0), mimo_rank, axis=1
        )
    else:
        mask_strict = mask_pos

    return jnp.where(mask_strict, x_segsum, -jnp.inf)


def ssd3_minimal_discrete_hybrid(
    X_c: jnp.ndarray,
    A_c: jnp.ndarray,
    B_c: jnp.ndarray,
    C_c: jnp.ndarray,
    last_chunk_indices: jnp.ndarray,
    initial_states: jnp.ndarray | None = None,
    reset_mask: jnp.ndarray | None = None,
    mimo_rank: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX Mamba-3 Selective State Space (SSD-3) scan over packed chunks.

    Implements the strict-causal discrete SSD-3 scan over packed chunks using
    XLA-lowered einsum contractions for intra-chunk state accumulation and
    state-to-output projection, and ``jax.lax.scan`` for ragged inter-chunk
    recurrence with sequence boundary resets.

    Args:
      X_c: (n_chunks, n_heads, block_len, d_head) - Packed value tensor
      A_c: (n_chunks, n_heads, block_len) - Log decay parameter (dt * A)
      B_c: (n_chunks, n_heads, block_len, d_state) - Rotated & scaled keys
      C_c: (n_chunks, n_heads, block_len, d_state) - Rotated queries
      last_chunk_indices: (num_seqs,) - Maps each sequence to its final chunk
      initial_states: (n_chunks, n_heads, d_head, d_state) - Optional initial
        states placed at sequence start chunks
      reset_mask: (n_chunks,) - Boolean mask indicating sequence start chunks
      mimo_rank: MIMO channel rank R (1 for SISO, >1 when
        block_len = chunk_size * R)

    Returns:
      Y: (n_chunks, n_heads, block_len, d_head) - Scan output (strict causal)
      final_states: (num_seqs, n_heads, d_head, d_state) - Recurrent states
    """
    n_chunks, n_heads, block_len, d_head = X_c.shape
    d_state = B_c.shape[-1]

    if initial_states is None:
        initial_states = jnp.zeros(
            (n_chunks, n_heads, d_head, d_state), dtype=jnp.float32
        )
    if reset_mask is None:
        reset_mask = jnp.zeros((n_chunks,), dtype=bool).at[0].set(True)

    A_c_f32 = A_c.astype(jnp.float32)

    if mimo_rank > 1:
        n_tok = block_len // mimo_rank
        X_r = X_c.reshape(n_chunks, n_heads, n_tok, mimo_rank, d_head)
        B_r = B_c.reshape(n_chunks, n_heads, n_tok, mimo_rank, d_state)
        C_r = C_c.reshape(n_chunks, n_heads, n_tok, mimo_rank, d_state)
        if A_c_f32.shape[-1] == block_len:
            A_tok = jnp.sum(
                A_c_f32.reshape(n_chunks, n_heads, n_tok, mimo_rank), axis=-1
            )
        else:
            A_tok = A_c_f32
        A_cumsum = jnp.cumsum(A_tok, axis=-1)

        # 1. Intra-chunk strict lower-triangular output (s < l)
        L_tok = jnp.exp(_segsum_strict_jax(A_tok, mimo_rank=1))
        cb = jnp.einsum(
            "chlrn,chsqn->chlsrq",
            C_r,
            B_r,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        intra_k = (cb * L_tok[:, :, :, :, None, None]).astype(X_c.dtype)
        Y_diag = jnp.einsum(
            "chlsrq,chsqp->chlrp",
            intra_k,
            X_r,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).reshape(n_chunks, n_heads, block_len, d_head)

        # 2. Intra-chunk state generation
        decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum).astype(X_c.dtype)
        X_scaled = X_r * decay_states[:, :, :, None, None]
        states_c = jnp.einsum(
            "chlrn,chlrp->chpn",
            B_r,
            X_scaled,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
    else:
        A_cumsum = jnp.cumsum(A_c_f32, axis=-1)

        # 1. Intra-chunk strict lower-triangular output (s < l)
        L = jnp.exp(_segsum_strict_jax(A_c_f32, mimo_rank=1))
        cb = jnp.einsum(
            "chln,chsn->chls",
            C_c,
            B_c,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        intra_k = (cb * L).astype(X_c.dtype)
        Y_diag = jnp.einsum(
            "chls,chsp->chlp",
            intra_k,
            X_c,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        # 2. Intra-chunk state generation
        decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum).astype(X_c.dtype)
        X_scaled = X_c * decay_states[..., None]
        states_c = jnp.einsum(
            "chln,chlp->chpn",
            B_c,
            X_scaled,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

    # 3. Inter-chunk recurrence across ragged sequences (JAX Scan)
    decay_chunk = jnp.exp(A_cumsum[..., -1])

    def scan_body(
        carry: jnp.ndarray,
        args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        states_c_c, decay_chunk_c, init_state_c, reset_c = args
        incoming_state = jnp.where(reset_c, init_state_c.astype(jnp.float32), carry)
        decay_exp = decay_chunk_c[:, None, None]
        next_state = incoming_state * decay_exp + states_c_c.astype(jnp.float32)
        return next_state, (incoming_state, next_state)

    scan_args = (states_c, decay_chunk, initial_states, reset_mask)
    init_carry = jnp.zeros((n_heads, d_head, d_state), dtype=jnp.float32)
    _, (states_incoming, states_outgoing) = jax.lax.scan(
        scan_body, init_carry, scan_args
    )

    # 4. Inter-chunk state to output conversion
    state_decay_out = jnp.exp(A_cumsum).astype(C_c.dtype)
    if mimo_rank > 1:
        n_tok = block_len // mimo_rank
        C_r_out = C_c.reshape(n_chunks, n_heads, n_tok, mimo_rank, d_state)
        C_scaled = C_r_out * state_decay_out[:, :, :, None, None]
        Y_off = jnp.einsum(
            "chlrn,chpn->chlrp",
            C_scaled,
            states_incoming.astype(C_c.dtype),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).reshape(n_chunks, n_heads, block_len, d_head)
    else:
        C_scaled = C_c * state_decay_out[..., None]
        Y_off = jnp.einsum(
            "chln,chpn->chlp",
            C_scaled,
            states_incoming.astype(C_c.dtype),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

    Y = (Y_diag + Y_off).astype(X_c.dtype)

    safe_last_chunk_indices = jnp.clip(last_chunk_indices, 0, n_chunks - 1)
    final_states = states_outgoing[safe_last_chunk_indices]
    return Y, final_states
