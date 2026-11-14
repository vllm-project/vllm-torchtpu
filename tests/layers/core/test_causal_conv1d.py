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
"""Unit tests for loop-free JAX ragged_causal_conv1d and mamba_utils."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core import causal_conv1d
from vllm_torchtpu.layers.core.causal_conv1d import ragged_causal_conv1d
from vllm_torchtpu.layers.core.mamba_utils import token_sequence_ids


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _causal_conv1d_numpy_reference(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array | None = None,
    has_initial_state: jax.Array | None = None,
    *,
    seq_lens: jax.Array | None = None,
    start_seq: int | None = None,
    activation: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent per-sequence, per-token float64 causal conv1d reference."""
    x_np = np.asarray(x.astype(jnp.float32), dtype=np.float64)
    orig_state_shape = conv_state.shape
    num_tokens, dim = x_np.shape
    history = orig_state_shape[1]
    kernel_size = history + 1

    state_f64 = np.asarray(conv_state.astype(jnp.float32), dtype=np.float64).reshape(
        orig_state_shape[0], history, dim
    )
    new_state_f64 = state_f64.copy()

    w_np = np.asarray(conv_weight.astype(jnp.float32), dtype=np.float64)
    w_kd = np.swapaxes(w_np[:, 0, :], 0, 1) if w_np.ndim == 3 else w_np
    assert w_kd.shape == (kernel_size, dim)

    bias_np = (
        np.asarray(conv_bias.astype(jnp.float32), dtype=np.float64)
        if conv_bias is not None
        else np.zeros((dim,), dtype=np.float64)
    )

    qsl_np = np.asarray(query_start_loc, dtype=np.int64)
    slots_np = np.asarray(state_indices, dtype=np.int64)
    num_seqs = len(slots_np)
    num_slots = orig_state_shape[0]

    if distribution is not None:
        num_valid_seqs = int(np.clip(int(np.asarray(distribution)[2]), 0, num_seqs))
        qsl_eff = qsl_np.copy()
        qsl_eff[num_valid_seqs + 1 :] = qsl_eff[num_valid_seqs]
    else:
        num_valid_seqs = num_seqs
        qsl_eff = qsl_np

    query_lens = np.maximum(qsl_eff[1:] - qsl_eff[:-1], 0)
    if has_initial_state is not None:
        has_init = np.asarray(has_initial_state, dtype=bool)
    elif seq_lens is not None:
        has_init = np.asarray(seq_lens, dtype=np.int64) > query_lens
    else:
        has_init = np.zeros((num_seqs,), dtype=bool)

    first_seq = 0 if start_seq is None else min(int(start_seq), num_valid_seqs)
    total_tokens = min(int(qsl_eff[-1]), num_tokens)
    out_f64 = np.zeros((num_tokens, dim), dtype=np.float64)

    for s in range(first_seq, num_valid_seqs):
        q_start = int(qsl_eff[s])
        q_end = min(int(qsl_eff[s + 1]), total_tokens)
        if q_start >= total_tokens or q_end <= q_start:
            continue
        slot = int(slots_np[s])
        valid_slot = 0 <= slot < num_slots
        window = (
            state_f64[slot].copy()
            if (has_init[s] and valid_slot)
            else np.zeros((history, dim), dtype=np.float64)
        )
        for t in range(q_start, q_end):
            full_taps = np.concatenate([window, x_np[t : t + 1]], axis=0)
            val = np.sum(full_taps * w_kd, axis=0) + bias_np
            if activation in ("silu", "swish"):
                val = _silu_np(val)
            out_f64[t] = val
            window = np.concatenate([window[1:], x_np[t : t + 1]], axis=0)
        if valid_slot and int(qsl_eff[s + 1]) <= num_tokens:
            new_state_f64[slot] = window

    out_ref = np.asarray(
        jnp.asarray(out_f64, dtype=jnp.float32).astype(x.dtype), dtype=np.float32
    )
    state_ref = np.asarray(
        jnp.asarray(new_state_f64.reshape(orig_state_shape), dtype=jnp.float32).astype(
            conv_state.dtype
        ),
        dtype=np.float32,
    )
    return out_ref, state_ref


def test_token_sequence_ids_handles_empty_interior_and_overflow_requests() -> None:
    # Seq 0: [0, 3), Seq 1: [3, 3) empty, Seq 2: [3, 6), Seq 3: [6, 10) at bucket=8.
    qsl = jnp.array([0, 3, 3, 6, 10], dtype=jnp.int32)
    seq_ids = token_sequence_ids(qsl, num_tokens=8, num_seqs=4)
    np.testing.assert_array_equal(
        np.asarray(seq_ids),
        np.array([0, 0, 0, 2, 2, 2, 3, 3], dtype=np.int32),
    )
    assert token_sequence_ids(qsl, num_tokens=0, num_seqs=4).shape == (0,)


def test_slot_zero_active_request_preserved_with_inactive_slots() -> None:
    """Active request in slot 0 must not be clobbered by inactive -1 or 0 slots."""
    rng = jax.random.key(11)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    dim, kernel_size, num_slots = 16, 4, 4

    x = jax.random.normal(k1, (8, dim), dtype=jnp.float32)
    conv_state = jax.random.normal(
        k2, (num_slots, kernel_size - 1, dim), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=jnp.float32)
    conv_bias = jax.random.normal(k4, (dim,), dtype=jnp.float32)

    # Seq 0 is active in slot 0 (5 tokens); Seq 1 is active in slot 2 (1 token);
    # Seq 2 is inactive with -1; Seq 3 is inactive aliasing slot 0.
    query_start_loc = jnp.array([0, 5, 6, 6, 6], dtype=jnp.int32)
    state_indices = jnp.array([0, 2, -1, 0], dtype=jnp.int32)
    distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
    has_initial_state = jnp.array([True, False, True, True], dtype=bool)

    out, new_state = ragged_causal_conv1d(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        activation="silu",
        kernel_size=kernel_size,
    )
    out_ref, state_ref = _causal_conv1d_numpy_reference(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        activation="silu",
    )

    np.testing.assert_allclose(np.asarray(out), out_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(new_state), state_ref, rtol=1e-5, atol=1e-5)
    # Untouched slots 1 and 3 must remain bit-identical.
    np.testing.assert_array_equal(np.asarray(new_state[1]), np.asarray(conv_state[1]))
    np.testing.assert_array_equal(np.asarray(new_state[3]), np.asarray(conv_state[3]))
    # Padded tail rows [6, 8) must be zero.
    np.testing.assert_array_equal(np.asarray(out[6:]), np.zeros((2, dim)))

    # Also verify slot 0 stays untouched when NO active sequence owns slot 0,
    # even if inactive tail entries point to slot 0 and -1.
    state_indices_no_slot0 = jnp.array([1, 2, 0, -1], dtype=jnp.int32)
    _, new_state_no_slot0 = ragged_causal_conv1d(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices_no_slot0,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )
    np.testing.assert_array_equal(
        np.asarray(new_state_no_slot0[0]), np.asarray(conv_state[0])
    )


@pytest.mark.parametrize("is_4d_state", [False, True])
@pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("x_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "swish"])
def test_ragged_causal_conv1d_3d_4d_dtypes_bias_and_initial_state_masking(
    is_4d_state: bool,
    state_dtype: jnp.dtype,
    x_dtype: jnp.dtype,
    use_bias: bool,
    activation: str | None,
) -> None:
    rng = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    # Include short prefill (len 2 < history=3, exercising partial window shift),
    # 1-token decode (len 1), and multi-token prefill (len 7 > history=3).
    num_tokens, dim, kernel_size, num_slots = 14, 12, 4, 5
    history = kernel_size - 1
    state_shape = (
        (num_slots, history, 1, dim) if is_4d_state else (num_slots, history, dim)
    )

    x = jax.random.normal(k1, (num_tokens, dim), dtype=x_dtype)
    # Large non-zero history so failing to mask has_initial_state=False is caught.
    conv_state = jax.random.normal(k2, state_shape, dtype=state_dtype) * 5.0
    conv_weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=x_dtype)
    conv_bias = jax.random.normal(k4, (dim,), dtype=x_dtype) if use_bias else None

    query_start_loc = jnp.array([0, 1, 3, 10, 10], dtype=jnp.int32)
    state_indices = jnp.array([0, 3, 1, -1], dtype=jnp.int32)
    distribution = jnp.array([1, 3, 3], dtype=jnp.int32)
    has_initial_state = jnp.array([True, False, True, False], dtype=bool)

    out, new_state = ragged_causal_conv1d(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        activation=activation,
        kernel_size=kernel_size,
    )
    out_ref, state_ref = _causal_conv1d_numpy_reference(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        activation=activation,
    )

    assert out.shape == x.shape
    assert out.dtype == x_dtype
    assert new_state.shape == state_shape
    assert new_state.dtype == state_dtype

    tol = 2e-2 if (x_dtype == jnp.bfloat16 or state_dtype == jnp.bfloat16) else 1e-5
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)), out_ref, rtol=tol, atol=tol
    )
    np.testing.assert_allclose(
        np.asarray(new_state.astype(jnp.float32)), state_ref, rtol=tol, atol=tol
    )

    # Must-fail control: flipping Seq 1 has_initial_state to True changes both
    # Seq 1's output and its carried partial window (since len=2 < history=3).
    out_unmasked, state_unmasked = ragged_causal_conv1d(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        jnp.array([True, True, True, False], dtype=bool),
        activation=activation,
        kernel_size=kernel_size,
    )
    assert not np.allclose(
        np.asarray(out[1:3].astype(jnp.float32)),
        np.asarray(out_unmasked[1:3].astype(jnp.float32)),
        atol=1e-2,
    )
    assert not np.allclose(
        np.asarray(new_state[3].astype(jnp.float32)),
        np.asarray(state_unmasked[3].astype(jnp.float32)),
        atol=1e-2,
    )


def test_2d_tap_major_weight_and_start_seq_restriction_even_when_k_equals_dim() -> None:
    """2D [K, D] and 3D [D, 1, K] weights agree when K == D == 4, with start_seq."""
    dim = kernel_size = 4
    rng = jax.random.key(77)
    k1, k2, k3 = jax.random.split(rng, 3)

    x = jax.random.normal(k1, (7, dim), dtype=jnp.float32)
    conv_state = jax.random.normal(k2, (4, kernel_size - 1, dim), dtype=jnp.float32)
    w_3d = jax.random.normal(k3, (dim, 1, kernel_size), dtype=jnp.float32)
    w_2d = jnp.swapaxes(w_3d[:, 0, :], 0, 1)

    qsl = jnp.array([0, 2, 7], dtype=jnp.int32)
    slots = jnp.array([0, 2], dtype=jnp.int32)
    seq_lens = jnp.array([5, 7], dtype=jnp.int32)

    out_3d, state_3d = ragged_causal_conv1d(
        x,
        conv_state,
        w_3d,
        None,
        qsl,
        slots,
        seq_lens=seq_lens,
        start_seq=1,
    )
    out_2d, state_2d = ragged_causal_conv1d(
        x,
        conv_state,
        w_2d,
        None,
        qsl,
        slots,
        seq_lens=seq_lens,
        start_seq=1,
    )
    out_ref, state_ref = _causal_conv1d_numpy_reference(
        x,
        conv_state,
        w_3d,
        None,
        qsl,
        slots,
        seq_lens=seq_lens,
        start_seq=1,
    )

    np.testing.assert_allclose(
        np.asarray(out_3d), np.asarray(out_2d), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(state_3d), np.asarray(state_2d), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(out_3d), out_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(state_3d), state_ref, rtol=1e-5, atol=1e-5)
    # Seq 0 (< start_seq=1) has zero output and untouched slot 0.
    np.testing.assert_array_equal(np.asarray(out_3d[:2]), np.zeros((2, dim)))
    np.testing.assert_array_equal(np.asarray(state_3d[0]), np.asarray(conv_state[0]))


def test_zero_active_tokens_returns_zeros_and_preserves_state() -> None:
    x = jnp.ones((4, 8), dtype=jnp.float32)
    conv_state = jnp.full((3, 3, 8), 9.0, dtype=jnp.float32)
    w = jnp.ones((8, 1, 4), dtype=jnp.float32)
    qsl = jnp.array([0, 0, 0], dtype=jnp.int32)
    slots = jnp.array([0, -1], dtype=jnp.int32)
    dist = jnp.array([0, 0, 0], dtype=jnp.int32)

    out, new_state = ragged_causal_conv1d(x, conv_state, w, None, qsl, slots, dist)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((4, 8)))
    np.testing.assert_array_equal(np.asarray(new_state), np.asarray(conv_state))


def test_ragged_causal_conv1d_validates_inputs() -> None:
    x = jnp.zeros((4, 8), dtype=jnp.float32)
    state = jnp.zeros((2, 3, 8), dtype=jnp.float32)
    w = jnp.zeros((8, 1, 4), dtype=jnp.float32)
    qsl = jnp.array([0, 4], dtype=jnp.int32)
    slots = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match="Expected 2D x"):
        ragged_causal_conv1d(x[:, None, :], state, w, None, qsl, slots)
    with pytest.raises(ValueError, match="conv_state trailing shape"):
        # Dim-first (S, D, K-1) layout must be rejected loudly.
        ragged_causal_conv1d(
            x, jnp.zeros((2, 8, 3), dtype=jnp.float32), w, None, qsl, slots
        )
    with pytest.raises(ValueError, match="does not match kernel_size=5"):
        ragged_causal_conv1d(x, state, w, None, qsl, slots, kernel_size=5)
    with pytest.raises(ValueError, match="Expected 3D conv_weight shape"):
        ragged_causal_conv1d(
            x, state, jnp.zeros((8, 2, 4), dtype=jnp.float32), None, qsl, slots
        )
    with pytest.raises(ValueError, match="Expected 2D conv_weight shape"):
        ragged_causal_conv1d(
            x, state, jnp.zeros((8, 4), dtype=jnp.float32), None, qsl, slots
        )
    with pytest.raises(ValueError, match="Expected conv_bias shape"):
        ragged_causal_conv1d(
            x, state, w, jnp.zeros((4,), dtype=jnp.float32), qsl, slots
        )
    with pytest.raises(ValueError, match="Unsupported activation"):
        ragged_causal_conv1d(x, state, w, None, qsl, slots, activation="gelu")


def test_tokamax_dispatch_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_tokamax_op(
        x: jax.Array, conv_state: jax.Array, *args: object, **kwargs: object
    ):
        calls.append(dict(kwargs))
        return x + 42.0, conv_state + 1.0

    monkeypatch.setattr(
        causal_conv1d, "_tokamax_ragged_causal_conv1d", _fake_tokamax_op
    )
    x = jnp.ones((2, 4), dtype=jnp.float32)
    state = jnp.zeros((2, 3, 4), dtype=jnp.float32)
    w = jnp.ones((4, 1, 4), dtype=jnp.float32)
    qsl = jnp.array([0, 2], dtype=jnp.int32)
    slots = jnp.array([0], dtype=jnp.int32)

    # Disabled by default when USE_TOKAMAX is False.
    monkeypatch.setattr(causal_conv1d, "USE_TOKAMAX", False)
    out_in_tree, _ = ragged_causal_conv1d(x, state, w, None, qsl, slots)
    assert len(calls) == 0
    assert not np.allclose(np.asarray(out_in_tree), 43.0)

    # Dispatches to Tokamax callable when USE_TOKAMAX is True.
    monkeypatch.setattr(causal_conv1d, "USE_TOKAMAX", True)
    out_tok, state_tok = ragged_causal_conv1d(
        x, state, w, None, qsl, slots, activation="silu", kernel_size=4
    )
    assert len(calls) == 1
    assert calls[0]["activation"] == "silu"
    assert calls[0]["kernel_size"] == 4
    np.testing.assert_array_equal(np.asarray(out_tok), np.full((2, 4), 43.0))
    np.testing.assert_array_equal(np.asarray(state_tok), np.ones((2, 3, 4)))
