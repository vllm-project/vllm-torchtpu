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
"""Correctness tests for Mamba-2 and Mamba-3 Selective State Space (SSD) scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import export

from vllm_torchtpu.kernels.ssd_scan import (
    chunk_scan_pallas,
    chunk_state_pallas,
    ssd_minimal_discrete_hybrid,
)
from vllm_torchtpu.kernels.ssd_scan import ssd_scan as ssd_scan_mod
from vllm_torchtpu.kernels.ssd_scan.ssd_scan import (
    _segsum_jax,
    _select_head_block_size,
)


def ssd_sequential_reference(
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
) -> tuple[jax.Array, jax.Array]:
    """Independent per-token sequential SSM recurrence reference in NumPy.

    Supports both Mamba-2 inclusive readout (`strict_causal=False`, `mimo_rank=1`)
    and Mamba-3 strict-causal readout (`strict_causal=True`, SISO or MIMO rank R)
    without calling `_segsum_jax` or chunked scan.
    """
    x_np = np.asarray(X_c.astype(jnp.float32), dtype=np.float32)
    a_np = np.asarray(A_c.astype(jnp.float32), dtype=np.float32)
    b_np = np.asarray(B_c.astype(jnp.float32), dtype=np.float32)
    c_np = np.asarray(C_c.astype(jnp.float32), dtype=np.float32)

    n_chunks, n_heads, block_len, d_head = x_np.shape
    n_groups = b_np.shape[1]
    d_state = b_np.shape[-1]
    n_tok = block_len // mimo_rank
    if n_groups < n_heads:
        repeats = n_heads // n_groups
        b_np = np.repeat(b_np, repeats, axis=1)
        c_np = np.repeat(c_np, repeats, axis=1)

    init_np = (
        None
        if initial_states is None
        else np.asarray(initial_states.astype(jnp.float32), dtype=np.float32)
    )

    if reset_mask is None:
        rst_np = np.zeros((n_chunks,), dtype=bool)
        rst_np[0] = True
    else:
        rst_np = np.asarray(reset_mask, dtype=bool)

    y_out = np.zeros_like(x_np, dtype=np.float32)
    states_outgoing = np.zeros((n_chunks, n_heads, d_head, d_state), dtype=np.float32)
    h = np.zeros((n_heads, d_head, d_state), dtype=np.float32)
    seq_idx = 0

    for c in range(n_chunks):
        if rst_np[c]:
            if init_np is None:
                h = np.zeros((n_heads, d_head, d_state), dtype=np.float32)
            else:
                safe_seq = min(seq_idx, init_np.shape[0] - 1)
                h = init_np[safe_seq].copy()
            seq_idx += 1
        for tok in range(n_tok):
            r_start = tok * mimo_rank
            r_end = r_start + mimo_rank
            if a_np.shape[-1] == n_tok:
                log_decay = a_np[c, :, tok]
            else:
                log_decay = np.sum(a_np[c, :, r_start:r_end], axis=-1)
            h = np.exp(log_decay)[:, None, None] * h

            if strict_causal:
                for r_idx in range(r_start, r_end):
                    y_out[c, :, r_idx, :] = np.einsum(
                        "hpn,hn->hp", h, c_np[c, :, r_idx, :]
                    )
                for r_idx in range(r_start, r_end):
                    h = h + x_np[c, :, r_idx, :, None] * b_np[c, :, r_idx, None, :]
            else:
                for r_idx in range(r_start, r_end):
                    h = h + x_np[c, :, r_idx, :, None] * b_np[c, :, r_idx, None, :]
                for r_idx in range(r_start, r_end):
                    y_out[c, :, r_idx, :] = np.einsum(
                        "hpn,hn->hp", h, c_np[c, :, r_idx, :]
                    )
        states_outgoing[c] = h.copy()

    safe_idx = np.maximum(np.asarray(last_chunk_indices, dtype=np.int64), 0)
    y_jax = jnp.asarray(y_out, dtype=jnp.float32).astype(X_c.dtype)
    final_states_jax = jnp.asarray(states_outgoing[safe_idx], dtype=jnp.float32)
    return y_jax, final_states_jax


def _make_ssd_inputs(
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    *,
    n_groups: int | None = None,
    dtype=jnp.float32,
):
    groups = n_heads if n_groups is None else n_groups
    n_chunks_per_seq = seqlen // block_len
    n_chunks = batch * n_chunks_per_seq

    rng = jax.random.key(42 + n_heads + groups + block_len)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head), dtype=dtype)
    A_c = -jnp.exp(
        jax.random.normal(k2, (n_chunks, n_heads, block_len), dtype=jnp.float32)
    )
    B_c = jax.random.normal(k3, (n_chunks, groups, block_len, d_state), dtype=dtype)
    C_c = jax.random.normal(k4, (n_chunks, groups, block_len, d_state), dtype=dtype)
    initial_states = jax.random.normal(
        k5, (batch, n_heads, d_head, d_state), dtype=dtype
    )

    reset_mask = np.zeros((n_chunks,), dtype=bool)
    for b in range(batch):
        reset_mask[b * n_chunks_per_seq] = True

    last_chunk_indices = jnp.array(
        [b * n_chunks_per_seq + n_chunks_per_seq - 1 for b in range(batch)],
        dtype=jnp.int32,
    )
    return X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, jnp.array(reset_mask)


def test_segsum_jax_against_naive_loop():
    x = jnp.array(
        [
            [-0.5, -1.25, -0.75, -2.0, -0.125],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
        ],
        dtype=jnp.float32,
    )
    actual = np.asarray(_segsum_jax(x))
    x_np = np.asarray(x)
    batch, length = x_np.shape
    expected = np.full((batch, length, length), -np.inf, dtype=np.float32)
    for b in range(batch):
        for i in range(length):
            expected[b, i, i] = 0.0
            for j in range(i):
                expected[b, i, j] = np.sum(x_np[b, j + 1 : i + 1])

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_segsum_jax_strict_causal_and_mimo_rank():
    x = jnp.array(
        [
            [-0.5, -1.25, -0.75, -2.0],
            [-0.1, -0.2, -0.3, -0.4],
        ],
        dtype=jnp.float32,
    )
    strict = np.asarray(jnp.exp(_segsum_jax(x, strict_causal=True)))
    x_np = np.asarray(x)
    for b in range(2):
        for i in range(4):
            for j in range(4):
                if i <= j:
                    assert strict[b, i, j] == 0.0
                else:
                    expected_val = float(np.exp(np.sum(x_np[b, j + 1 : i + 1])))
                    np.testing.assert_allclose(
                        strict[b, i, j], expected_val, rtol=1e-6, atol=1e-6
                    )

    # MIMO rank=2 over 3 tokens (T=6), decay placed at sub-channel 0 of each token
    a_tok = np.array([[-0.3, -0.7, -1.1]], dtype=np.float32)
    a_exp = np.zeros((1, 3, 2), dtype=np.float32)
    a_exp[:, :, 0] = a_tok
    L_mimo = np.asarray(
        jnp.exp(
            _segsum_jax(
                jnp.asarray(a_exp.reshape(1, 6)), strict_causal=True, mimo_rank=2
            )
        )
    )
    for l_tok in range(3):
        for s_tok in range(3):
            block = L_mimo[0, l_tok * 2 : (l_tok + 1) * 2, s_tok * 2 : (s_tok + 1) * 2]
            if l_tok <= s_tok:
                np.testing.assert_array_equal(block, np.zeros((2, 2), dtype=np.float32))
            else:
                expected_decay = float(np.exp(np.sum(a_tok[0, s_tok + 1 : l_tok + 1])))
                np.testing.assert_allclose(
                    block, np.full((2, 2), expected_decay), rtol=1e-6, atol=1e-6
                )

    with pytest.raises(ValueError, match="Expected mimo_rank >= 1"):
        _segsum_jax(x, mimo_rank=0)
    with pytest.raises(ValueError, match="divisible by mimo_rank"):
        _segsum_jax(x, mimo_rank=3)


def test_ssd_scan_validates_input_shapes():
    n_chunks, n_heads, block_len, d_head, d_state = 2, 4, 8, 16, 8
    X_c = jnp.zeros((n_chunks, n_heads, block_len, d_head), dtype=jnp.float32)
    A_c = jnp.zeros((n_chunks, n_heads, block_len), dtype=jnp.float32)
    B_c = jnp.zeros((n_chunks, n_heads, block_len, d_state), dtype=jnp.float32)
    C_c = jnp.zeros((n_chunks, n_heads, block_len, d_state), dtype=jnp.float32)
    decay_states = jnp.zeros((n_chunks, n_heads, 1, block_len), dtype=jnp.float32)
    states = jnp.zeros((n_chunks, n_heads, d_head, d_state), dtype=jnp.float32)
    last_chunk_indices = jnp.array([0, 1], dtype=jnp.int32)

    # chunk_state_pallas shape and positive-dimension validation
    with pytest.raises(ValueError, match="Expected positive 4D X_c"):
        chunk_state_pallas(B_c, decay_states, X_c[0], interpret=True)
    with pytest.raises(ValueError, match="Expected positive 4D X_c"):
        chunk_state_pallas(B_c[:0], decay_states[:0], X_c[:0], interpret=True)
    with pytest.raises(ValueError, match="Expected positive 4D B_c"):
        chunk_state_pallas(B_c[:, :3], decay_states, X_c, interpret=True)
    with pytest.raises(ValueError, match="Expected decay_states shape"):
        chunk_state_pallas(B_c, jnp.swapaxes(decay_states, -1, -2), X_c, interpret=True)

    # chunk_scan_pallas shape and positive-dimension validation
    with pytest.raises(ValueError, match="Expected positive 4D C_c"):
        chunk_scan_pallas(C_c[0], states, interpret=True)
    with pytest.raises(ValueError, match="Expected positive 4D C_c"):
        chunk_scan_pallas(C_c[:0], states[:0], interpret=True)
    with pytest.raises(ValueError, match="Expected positive 4D states"):
        chunk_scan_pallas(C_c[:, :3], states, interpret=True)

    # ssd_minimal_discrete_hybrid shape and positive-dimension validation
    with pytest.raises(ValueError, match="Expected positive 4D X_c"):
        ssd_minimal_discrete_hybrid(
            X_c[:0], A_c[:0], B_c[:0], C_c[:0], last_chunk_indices, interpret=True
        )
    with pytest.raises(ValueError, match="Expected A_c shape"):
        ssd_minimal_discrete_hybrid(
            X_c, A_c[:, :2], B_c, C_c, last_chunk_indices, interpret=True
        )
    with pytest.raises(ValueError, match="Expected positive 4D B_c"):
        ssd_minimal_discrete_hybrid(
            X_c, A_c, B_c[:, :3], C_c[:, :3], last_chunk_indices, interpret=True
        )
    with pytest.raises(ValueError, match="Expected C_c shape"):
        ssd_minimal_discrete_hybrid(
            X_c, A_c, B_c, C_c[:, :2], last_chunk_indices, interpret=True
        )
    with pytest.raises(ValueError, match="Expected non-empty 1D last_chunk_indices"):
        ssd_minimal_discrete_hybrid(
            X_c, A_c, B_c, C_c, last_chunk_indices[:, None], interpret=True
        )
    with pytest.raises(ValueError, match="Expected non-empty 1D last_chunk_indices"):
        ssd_minimal_discrete_hybrid(
            X_c, A_c, B_c, C_c, jnp.zeros((0,), dtype=jnp.int32), interpret=True
        )
    with pytest.raises(ValueError, match="Expected initial_states shape"):
        ssd_minimal_discrete_hybrid(
            X_c,
            A_c,
            B_c,
            C_c,
            last_chunk_indices,
            initial_states=states[:, :2],
            interpret=True,
        )
    with pytest.raises(ValueError, match="Expected reset_mask shape"):
        ssd_minimal_discrete_hybrid(
            X_c,
            A_c,
            B_c,
            C_c,
            last_chunk_indices,
            reset_mask=jnp.ones((n_chunks + 1,), dtype=bool),
            interpret=True,
        )
    with pytest.raises(ValueError, match="Expected mimo_rank >= 1"):
        ssd_minimal_discrete_hybrid(
            X_c, A_c, B_c, C_c, last_chunk_indices, mimo_rank=0, interpret=True
        )
    with pytest.raises(ValueError, match="divisible by mimo_rank"):
        ssd_minimal_discrete_hybrid(
            X_c, A_c, B_c, C_c, last_chunk_indices, mimo_rank=3, interpret=True
        )
    with pytest.raises(ValueError, match="Expected A_c shape in"):
        ssd_minimal_discrete_hybrid(
            X_c,
            jnp.zeros((n_chunks, n_heads, 3), dtype=jnp.float32),
            B_c,
            C_c,
            last_chunk_indices,
            mimo_rank=4,
            interpret=True,
        )


@pytest.mark.parametrize(
    ("batch", "seqlen", "block_len", "n_heads", "d_head", "d_state"),
    [
        (2, 16, 4, 4, 8, 8),
        (2, 128, 32, 4, 16, 8),
        (4, 256, 64, 8, 32, 16),
        (2, 64, 16, 6, 16, 8),
        (2, 64, 16, 12, 16, 8),
        (2, 32, 16, 1, 16, 8),
    ],
)
def test_ssd_scan_correctness(
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
):
    X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            batch, seqlen, block_len, n_heads, d_head, d_state, dtype=jnp.float32
        )
    )

    y_ref, final_state_ref = ssd_sequential_reference(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask
    )
    y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )

    assert final_state_hybrid.dtype == jnp.float32
    np.testing.assert_allclose(y_hybrid, y_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        final_state_hybrid, final_state_ref, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    ("n_heads", "n_groups"),
    [
        (8, 2),
        (12, 2),
        (6, 1),
    ],
)
def test_ssd_scan_grouped_bc_projections(
    dtype,
    n_heads: int,
    n_groups: int,
):
    X_c, A_c, B_grouped, C_grouped, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            2,
            64,
            16,
            n_heads,
            16,
            8,
            n_groups=n_groups,
            dtype=dtype,
        )
    )
    repeats = n_heads // n_groups
    B_rep = jnp.repeat(B_grouped, repeats, axis=1)
    C_rep = jnp.repeat(C_grouped, repeats, axis=1)

    y_grouped, state_grouped = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_grouped,
        C_grouped,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )
    y_rep, state_rep = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_rep,
        C_rep,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )
    y_ref, state_ref = ssd_sequential_reference(
        X_c,
        A_c,
        B_grouped,
        C_grouped,
        last_chunk_indices,
        initial_states,
        reset_mask,
    )

    np.testing.assert_allclose(
        np.asarray(state_grouped), np.asarray(state_rep), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(y_grouped.astype(jnp.float32)),
        np.asarray(y_rep.astype(jnp.float32)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(state_grouped), np.asarray(state_ref), rtol=1e-4, atol=1e-4
    )
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(
        np.asarray(y_grouped.astype(jnp.float32)),
        np.asarray(y_ref.astype(jnp.float32)),
        rtol=tol,
        atol=tol,
    )


def test_ssd_scan_defaults_start_one_sequence_from_zero_state():
    X_c, A_c, B_c, C_c, last_chunk_indices, _, _ = _make_ssd_inputs(1, 64, 16, 4, 16, 8)
    for strict_causal in (False, True):
        y_ref, final_state_ref = ssd_sequential_reference(
            X_c,
            A_c,
            B_c,
            C_c,
            last_chunk_indices,
            strict_causal=strict_causal,
        )
        y, final_state = ssd_minimal_discrete_hybrid(
            X_c,
            A_c,
            B_c,
            C_c,
            last_chunk_indices,
            strict_causal=strict_causal,
            interpret=True,
        )
        np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(final_state, final_state_ref, rtol=1e-4, atol=1e-4)


def test_ssd_scan_ragged_tail_padding_sentinels_and_checkpoints():
    """Verify ragged chunk counts, sub-chunk tail padding, `initial_states=None`
    multi-sequence resets, `-1` sentinels, and intermediate checkpoint gathering.
    """
    block_len, n_heads, n_groups, d_head, d_state = 8, 6, 2, 8, 8
    seq_token_lens = [5, 19, 16]
    seq_chunk_counts = [(L + block_len - 1) // block_len for L in seq_token_lens]
    n_chunks = sum(seq_chunk_counts)

    rng = jax.random.key(123)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    X_raw = np.array(
        jax.random.normal(
            k1, (n_chunks, n_heads, block_len, d_head), dtype=jnp.float32
        ),
        copy=True,
    )
    A_raw = np.array(
        -jnp.exp(
            jax.random.normal(k2, (n_chunks, n_heads, block_len), dtype=jnp.float32)
        ),
        copy=True,
    )
    B_raw = np.array(
        jax.random.normal(
            k3, (n_chunks, n_groups, block_len, d_state), dtype=jnp.float32
        ),
        copy=True,
    )
    C_raw = np.array(
        jax.random.normal(
            k4, (n_chunks, n_groups, block_len, d_state), dtype=jnp.float32
        ),
        copy=True,
    )
    init_states = jax.random.normal(
        k5, (len(seq_token_lens), n_heads, d_head, d_state), dtype=jnp.float32
    )

    reset_mask = np.zeros((n_chunks,), dtype=bool)
    seq_last_chunks: list[int] = []
    chunk_cursor = 0
    for seq_len, n_seq_chunks in zip(seq_token_lens, seq_chunk_counts):
        reset_mask[chunk_cursor] = True
        last_c = chunk_cursor + n_seq_chunks - 1
        seq_last_chunks.append(last_c)
        rem = seq_len % block_len
        if rem != 0:
            X_raw[last_c, :, rem:, :] = 0.0
            A_raw[last_c, :, rem:] = 0.0
        chunk_cursor += n_seq_chunks

    X_c = jnp.asarray(X_raw)
    A_c = jnp.asarray(A_raw)
    B_c = jnp.asarray(B_raw)
    C_c = jnp.asarray(C_raw)
    reset_mask_jax = jnp.asarray(reset_mask)

    checkpoint_and_final_indices = jnp.array([0, 1, 2, 3, 5, -1, -1], dtype=jnp.int32)

    y_init, states_init = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        checkpoint_and_final_indices,
        initial_states=init_states,
        reset_mask=reset_mask_jax,
        interpret=True,
    )
    y_init_ref, states_init_ref = ssd_sequential_reference(
        X_c,
        A_c,
        B_c,
        C_c,
        checkpoint_and_final_indices,
        initial_states=init_states,
        reset_mask=reset_mask_jax,
    )
    np.testing.assert_allclose(y_init, y_init_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(states_init, states_init_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(states_init[-1], states_init[0], rtol=1e-6, atol=1e-6)

    init_states_padded = jnp.pad(
        init_states,
        ((0, n_chunks - len(seq_token_lens)), (0, 0), (0, 0), (0, 0)),
        constant_values=99.0,
    )
    y_pad_init, states_pad_init = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        checkpoint_and_final_indices,
        initial_states=init_states_padded,
        reset_mask=reset_mask_jax,
        interpret=True,
    )
    np.testing.assert_allclose(y_pad_init, y_init_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(states_pad_init, states_init_ref, rtol=1e-4, atol=1e-4)

    b_rep = np.repeat(B_raw, n_heads // n_groups, axis=1)
    h_seq0 = np.asarray(init_states[0], dtype=np.float32).copy()
    for t in range(seq_token_lens[0]):
        h_seq0 = (
            np.exp(A_raw[0, :, t])[:, None, None] * h_seq0
            + X_raw[0, :, t, :, None] * b_rep[0, :, t, None, :]
        )
    np.testing.assert_allclose(states_init[0], h_seq0, rtol=1e-4, atol=1e-4)

    last_only = jnp.asarray(seq_last_chunks, dtype=jnp.int32)
    y_zero, states_zero = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        last_only,
        initial_states=None,
        reset_mask=reset_mask_jax,
        interpret=True,
    )
    y_zero_ref, states_zero_ref = ssd_sequential_reference(
        X_c,
        A_c,
        B_c,
        C_c,
        last_only,
        initial_states=None,
        reset_mask=reset_mask_jax,
    )
    np.testing.assert_allclose(y_zero, y_zero_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(states_zero, states_zero_ref, rtol=1e-4, atol=1e-4)


def test_ssd_scan_vmem_derived_head_block_and_mosaic_lowering():
    """Verify `_select_head_block_size` queries `pltpu.get_tpu_info()` when
    `not interpret` and lowers cleanly to Mosaic TPU MLIR across shapes.
    """

    class _FakeTpuDevice:
        def __init__(self, device_kind: str):
            self.device_kind = device_kind
            self.platform = "tpu"
            self.num_cores = 1

    mesh_16m = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=_FakeTpuDevice("TPU v4")
    )
    mesh_128m = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=_FakeTpuDevice("TPU v6 lite")
    )

    with jax.sharding.use_abstract_mesh(mesh_16m):
        hb_16m = _select_head_block_size(
            64, 64, 512, 128, 256, include_decay_tile=True, interpret=False
        )
        hb_grouped = _select_head_block_size(
            12, 2, 64, 64, 128, include_decay_tile=True, interpret=False
        )
    with jax.sharding.use_abstract_mesh(mesh_128m):
        hb_128m = _select_head_block_size(
            64, 64, 512, 128, 256, include_decay_tile=True, interpret=False
        )

    assert hb_16m == 4
    assert hb_grouped == 6
    assert hb_128m > hb_16m

    with jax.sharding.use_abstract_mesh(mesh_128m):
        jitted = jax.jit(ssd_minimal_discrete_hybrid)
        exported = export.export(jitted, platforms=("tpu",))(
            jax.ShapeDtypeStruct((3, 12, 64, 64), jnp.bfloat16),
            jax.ShapeDtypeStruct((3, 12, 64), jnp.float32),
            jax.ShapeDtypeStruct((3, 2, 64, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((3, 2, 64, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((2, 12, 64, 128), jnp.float32),
            jax.ShapeDtypeStruct((3,), jnp.bool_),
        )
        assert "ssd_chunk_state_pallas" in exported.mlir_module()
        assert "ssd_chunk_scan_pallas" in exported.mlir_module()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("mimo_rank", [1, 4])
@pytest.mark.parametrize("per_token_a", [True, False])
def test_ssd_scan_strict_causal_siso_and_mimo_correctness(
    dtype, mimo_rank: int, per_token_a: bool
):
    if mimo_rank == 1 and not per_token_a:
        pytest.skip("mimo_rank=1 has identical per-token and block_len A_c shape")

    seq_chunk_counts = [1, 3, 2]
    n_chunks = sum(seq_chunk_counts)
    n_tok = 8
    block_len = n_tok * mimo_rank
    n_heads, d_head, d_state = 4, 16, 8

    rng = jax.random.key(101 + mimo_rank)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head), dtype=dtype)
    A_tok = (
        -jax.nn.softplus(
            jax.random.normal(k2, (n_chunks, n_heads, n_tok), dtype=jnp.float32)
        )
        * 0.15
    )
    if per_token_a or mimo_rank == 1:
        A_c = A_tok
    else:
        weights = jnp.linspace(0.1, 0.4, mimo_rank, dtype=jnp.float32)
        weights = weights / jnp.sum(weights)
        A_c = (A_tok[..., None] * weights).reshape(n_chunks, n_heads, block_len)

    B_c = jax.random.normal(k3, (n_chunks, n_heads, block_len, d_state), dtype=dtype)
    C_c = jax.random.normal(k4, (n_chunks, n_heads, block_len, d_state), dtype=dtype)
    initial_states = jax.random.normal(
        k5, (len(seq_chunk_counts), n_heads, d_head, d_state), dtype=jnp.float32
    )

    reset_mask = np.zeros((n_chunks,), dtype=bool)
    last_chunks = []
    cursor = 0
    for count in seq_chunk_counts:
        reset_mask[cursor] = True
        last_chunks.append(cursor + count - 1)
        cursor += count
    last_chunks.append(-1)  # inactive padded sequence slot
    last_chunk_indices = jnp.array(last_chunks, dtype=jnp.int32)
    reset_mask_jax = jnp.array(reset_mask)

    y_ref, final_state_ref = ssd_sequential_reference(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask_jax,
        strict_causal=True,
        mimo_rank=mimo_rank,
    )
    y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask_jax,
        strict_causal=True,
        mimo_rank=mimo_rank,
        interpret=True,
    )

    assert y_hybrid.dtype == dtype
    assert final_state_hybrid.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(final_state_hybrid[: len(seq_chunk_counts)]),
        np.asarray(final_state_ref[: len(seq_chunk_counts)]),
        rtol=1e-4,
        atol=1e-4,
    )
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(
        np.asarray(y_hybrid.astype(jnp.float32)),
        np.asarray(y_ref.astype(jnp.float32)),
        rtol=tol,
        atol=tol,
    )


def test_ssd_scan_tokamax_seam_dispatch(monkeypatch):
    X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(2, 16, 8, 2, 8, 8, dtype=jnp.float32)
    )
    calls: list[dict[str, object]] = []

    def _fake_tokamax_ssd(
        x_arg,
        a_arg,
        b_arg,
        c_arg,
        last_arg,
        init_arg=None,
        reset_arg=None,
        *,
        strict_causal: bool = False,
        mimo_rank: int = 1,
        interpret: bool = False,
    ):
        calls.append(
            {
                "strict_causal": strict_causal,
                "mimo_rank": mimo_rank,
                "interpret": interpret,
            }
        )
        return jnp.full_like(x_arg, 9.0), jnp.full(
            (last_arg.shape[0], x_arg.shape[1], x_arg.shape[3], b_arg.shape[3]),
            5.0,
            dtype=jnp.float32,
        )

    monkeypatch.setattr(ssd_scan_mod, "_tokamax_ssd_scan", _fake_tokamax_ssd)
    monkeypatch.setattr(ssd_scan_mod, "USE_TOKAMAX", False)
    y_native, _ = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )
    assert len(calls) == 0
    assert not jnp.all(y_native == 9.0)

    monkeypatch.setattr(ssd_scan_mod, "USE_TOKAMAX", True)
    y_tok, s_tok = ssd_minimal_discrete_hybrid(
        X_c,
        A_c[:, :, ::2],
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=True,
        mimo_rank=2,
        interpret=True,
    )
    assert len(calls) == 1
    assert calls[0] == {"strict_causal": True, "mimo_rank": 2, "interpret": True}
    assert jnp.all(y_tok == 9.0)
    assert jnp.all(s_tok == 5.0)


@pytest.mark.parametrize(
    ("batch", "seqlen", "block_len", "n_heads", "d_head", "d_state"),
    [
        (2, 64, 16, 6, 16, 8),
        (2, 256, 64, 16, 64, 128),
    ],
)
def test_ssd_scan_bfloat16_inputs_preserve_float32_state_carry(
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
):
    X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            batch, seqlen, block_len, n_heads, d_head, d_state, dtype=jnp.bfloat16
        )
    )

    A_cumsum = jnp.cumsum(A_c, axis=-1)
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)[:, :, None, :]
    states_c = chunk_state_pallas(B_c, decay_states, X_c, interpret=True)
    assert states_c.dtype == jnp.float32

    y_out_pallas = chunk_scan_pallas(C_c, states_c, interpret=True)
    assert y_out_pallas.dtype == jnp.bfloat16
    y_out_expected = jnp.einsum(
        "chln,chpn->chlp",
        C_c.astype(jnp.float32),
        states_c,
        precision=jax.lax.Precision.HIGHEST,
    ).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(y_out_pallas.astype(jnp.float32)),
        np.asarray(y_out_expected.astype(jnp.float32)),
        rtol=1e-2,
        atol=1e-2,
    )

    y_ref, final_state_ref = ssd_sequential_reference(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask
    )
    y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )

    assert y_hybrid.dtype == jnp.bfloat16
    assert final_state_hybrid.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(final_state_hybrid),
        np.asarray(final_state_ref),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(y_hybrid.astype(jnp.float32)),
        np.asarray(y_ref.astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("strict_causal", [False, True])
@pytest.mark.parametrize("mimo_rank", [1, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    ("batch", "seqlen", "block_len", "n_heads", "n_groups", "d_head", "d_state"),
    [
        (2, 128, 32, 4, 4, 16, 8),
        (2, 128, 64, 6, 6, 64, 128),
        (2, 128, 64, 12, 2, 64, 128),
        (2, 256, 64, 16, 4, 64, 128),
    ],
)
def test_ssd_scan_correctness_tpu(
    strict_causal: bool,
    mimo_rank: int,
    dtype,
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    n_groups: int,
    d_head: int,
    d_state: int,
):
    if mimo_rank > 1 and not strict_causal:
        pytest.skip("MIMO mode is used with strict_causal=True")
    X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            batch,
            seqlen,
            block_len,
            n_heads,
            d_head,
            d_state,
            n_groups=n_groups,
            dtype=dtype,
        )
    )
    if mimo_rank > 1:
        A_c = A_c[:, :, ::mimo_rank]

    y_ref, final_state_ref = ssd_sequential_reference(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=strict_causal,
        mimo_rank=mimo_rank,
    )
    jitted_ssd = jax.jit(
        ssd_minimal_discrete_hybrid,
        static_argnames=("strict_causal", "mimo_rank"),
    )
    y_hybrid, final_state_hybrid = jitted_ssd(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=strict_causal,
        mimo_rank=mimo_rank,
    )

    assert y_hybrid.dtype == dtype
    assert final_state_hybrid.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(final_state_hybrid),
        np.asarray(final_state_ref),
        rtol=1e-3,
        atol=1e-3,
    )
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-3
    np.testing.assert_allclose(
        np.asarray(y_hybrid.astype(jnp.float32)),
        np.asarray(y_ref.astype(jnp.float32)),
        rtol=tol,
        atol=tol,
    )
