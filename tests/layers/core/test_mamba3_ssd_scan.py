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
"""Correctness tests for Mamba-3 SSD-3 JAX scan kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vllm  # noqa: F401

from vllm_torchtpu.layers.core.mamba3_ssd_scan import (
    _segsum_strict_jax,
    ssd3_minimal_discrete_hybrid,
)


def ssd3_sequential_reference(
    X_c: jnp.ndarray,
    A_c: jnp.ndarray,
    B_c: jnp.ndarray,
    C_c: jnp.ndarray,
    last_chunk_indices: jnp.ndarray,
    initial_states: jnp.ndarray,
    reset_mask: jnp.ndarray,
    mimo_rank: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Step-by-step sequential recurrence reference for Mamba-3 strict causal scan."""
    n_chunks, n_heads, block_len, d_head = X_c.shape
    d_state = B_c.shape[-1]
    n_tok = block_len // mimo_rank

    Y_out = np.zeros((n_chunks, n_heads, block_len, d_head), dtype=np.float32)
    states_outgoing = np.zeros((n_chunks, n_heads, d_head, d_state), dtype=np.float32)

    X_np = np.asarray(X_c, dtype=np.float32)
    A_np = np.asarray(A_c, dtype=np.float32)
    B_np = np.asarray(B_c, dtype=np.float32)
    C_np = np.asarray(C_c, dtype=np.float32)
    init_np = np.asarray(initial_states, dtype=np.float32)
    reset_np = np.asarray(reset_mask, dtype=bool)

    h = np.zeros((n_heads, d_head, d_state), dtype=np.float32)
    for c_idx in range(n_chunks):
        if reset_np[c_idx]:
            h = init_np[c_idx].copy()

        for tok_idx in range(n_tok):
            r_start = tok_idx * mimo_rank
            r_end = r_start + mimo_rank
            # Decay occurs at the first rank channel of each token
            decay_tok = np.exp(np.sum(A_np[c_idx, :, r_start:r_end], axis=-1))
            h = h * decay_tok[:, None, None]

            # Strict causal readout (before writing current token's channels)
            for r_idx in range(r_start, r_end):
                Y_out[c_idx, :, r_idx, :] = np.einsum(
                    "hpn,hn->hp", h, C_np[c_idx, :, r_idx, :]
                )

            # Write all MIMO rank channels for current token into state
            for r_idx in range(r_start, r_end):
                h = h + X_np[c_idx, :, r_idx, :, None] * B_np[c_idx, :, r_idx, None, :]

        states_outgoing[c_idx] = h.copy()

    final_states = states_outgoing[np.asarray(last_chunk_indices)]
    return jnp.asarray(Y_out, dtype=X_c.dtype), jnp.asarray(final_states)


def test_import_does_not_override_global_matmul_precision() -> None:
    assert jax.config.jax_default_matmul_precision is None


@pytest.mark.parametrize(
    ("batch", "seqlen", "block_len", "n_heads", "d_head", "d_state"),
    [
        (2, 64, 16, 4, 16, 8),
        (3, 128, 32, 8, 32, 16),
    ],
)
def test_ssd3_scan_siso_matches_sequential_reference(
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
) -> None:
    n_chunks_per_seq = seqlen // block_len
    n_chunks = batch * n_chunks_per_seq

    rng = jax.random.key(42)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head))
    A_c = -jnp.exp(jax.random.normal(k2, (n_chunks, n_heads, block_len))) * 0.1
    B_c = jax.random.normal(k3, (n_chunks, n_heads, block_len, d_state))
    C_c = jax.random.normal(k4, (n_chunks, n_heads, block_len, d_state))
    init_states_per_batch = jax.random.normal(k5, (batch, n_heads, d_head, d_state))

    initial_states = jnp.zeros((n_chunks, n_heads, d_head, d_state))
    reset_mask = np.zeros((n_chunks,), dtype=bool)
    for b in range(batch):
        initial_states = initial_states.at[b * n_chunks_per_seq].set(
            init_states_per_batch[b]
        )
        reset_mask[b * n_chunks_per_seq] = True
    reset_mask_jax = jnp.array(reset_mask)
    last_chunk_indices = jnp.array(
        [b * n_chunks_per_seq + n_chunks_per_seq - 1 for b in range(batch)],
        dtype=jnp.int32,
    )

    y_ref, final_state_ref = ssd3_sequential_reference(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask_jax
    )
    y_hybrid, final_state_hybrid = ssd3_minimal_discrete_hybrid(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask_jax
    )

    np.testing.assert_allclose(y_hybrid, y_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        final_state_hybrid, final_state_ref, rtol=1e-4, atol=1e-4
    )


def test_ssd3_scan_mimo_matches_sequential_reference() -> None:
    batch, n_chunks_per_seq, chunk_tokens, mimo_rank = 2, 2, 8, 4
    n_chunks = batch * n_chunks_per_seq
    block_len = chunk_tokens * mimo_rank
    n_heads, d_head, d_state = 4, 8, 8

    rng = jax.random.key(99)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head))
    A_tok = (
        -jax.nn.softplus(jax.random.normal(k2, (n_chunks, n_heads, chunk_tokens))) * 0.1
    )
    A_zeros = jnp.zeros(
        (n_chunks, n_heads, chunk_tokens, mimo_rank - 1), dtype=jnp.float32
    )
    A_c = jnp.concatenate([A_tok[..., None], A_zeros], axis=-1).reshape(
        n_chunks, n_heads, block_len
    )
    B_c = jax.random.normal(k3, (n_chunks, n_heads, block_len, d_state))
    C_c = jax.random.normal(k4, (n_chunks, n_heads, block_len, d_state))
    init_states_per_batch = jax.random.normal(k5, (batch, n_heads, d_head, d_state))

    initial_states = jnp.zeros((n_chunks, n_heads, d_head, d_state))
    reset_mask = np.zeros((n_chunks,), dtype=bool)
    for b in range(batch):
        initial_states = initial_states.at[b * n_chunks_per_seq].set(
            init_states_per_batch[b]
        )
        reset_mask[b * n_chunks_per_seq] = True
    reset_mask_jax = jnp.array(reset_mask)
    last_chunk_indices = jnp.array(
        [b * n_chunks_per_seq + n_chunks_per_seq - 1 for b in range(batch)],
        dtype=jnp.int32,
    )

    y_ref, final_state_ref = ssd3_sequential_reference(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask_jax,
        mimo_rank=mimo_rank,
    )
    y_hybrid, final_state_hybrid = ssd3_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask_jax,
        mimo_rank=mimo_rank,
    )

    np.testing.assert_allclose(y_hybrid, y_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        final_state_hybrid, final_state_ref, rtol=1e-4, atol=1e-4
    )


def test_segsum_strict_excludes_diagonal() -> None:
    x = jnp.ones((2, 4), dtype=jnp.float32)
    L = jnp.exp(_segsum_strict_jax(x))
    # Diagonal and upper triangle must be strictly 0.0
    for i in range(4):
        for j in range(i, 4):
            assert float(L[0, i, j]) == 0.0
