# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused KDA decode kernel.

``decode_kda`` does in one Pallas call what main splits across two custom ops:
the paged short convolution plus SiLU, then the single-token recurrent update.
Fusing them means the kernel owns two paged state pools at once, and most of what
can go wrong is in that ownership rather than in the arithmetic -- which slot it
reads, which it writes, and what it does with a request that has no state yet.
So alongside the numerical checks there are explicit tests for the null block,
for padded request rows, and for a sequence's first step.

The numerical reference (:func:`_reference_step`) is float64 and runs one
sequence at a time. It deliberately reproduces two of the kernel's roundings --
the post-SiLU cast back to bfloat16, and the bfloat16 output -- so the
tolerances measure the kernel's own error rather than the dtype's.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.kimi_k3 import ragged_kda
from vllm_torchtpu.kernels.kimi_k3.decode_kda import decode_kda
from vllm_torchtpu.layers.vllm.custom_ops.kda_attention_op import \
    kimi_short_conv_scan

HEADS = 2
HEAD_DIM = 128
KERNEL_SIZE = 4
PROJECTION = HEADS * HEAD_DIM
MIXED_DIM = 3 * PROJECTION
SCALE = HEAD_DIM**-0.5
EPS = 1e-5

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _build(num_live,
           bucket,
           *,
           seed=0,
           carry=True,
           conv_state_dim_first=False,
           context_len=128):
    """One decode step: ``num_live`` one-token requests padded to ``bucket``.

    ``carry`` sets whether the live requests have context behind them. Slot 0 is
    the reserved null block, so live requests take slots 1..num_live.
    """
    rng = np.random.default_rng(seed)
    num_slots = bucket + 1

    def activation(width):
        return jnp.asarray(rng.standard_normal((bucket, width)), jnp.bfloat16)

    cumulative = list(range(num_live + 1)) + [num_live] * (bucket - num_live)
    slots = [i + 1 for i in range(num_live)] + [0] * (bucket - num_live)
    query_len, prior = 1, (context_len if carry else 0)
    lengths = [query_len + prior] * num_live + [0] * (bucket - num_live)

    conv_tail = ((MIXED_DIM, KERNEL_SIZE - 1) if conv_state_dim_first else
                 (KERNEL_SIZE - 1, MIXED_DIM))
    conv_state = jnp.asarray(
        rng.standard_normal((num_slots, ) + conv_tail) * 0.1, jnp.bfloat16)
    conv_weight = jnp.asarray(
        rng.standard_normal((MIXED_DIM, 1, KERNEL_SIZE)) * 0.3, jnp.bfloat16)
    pool = jnp.asarray(
        rng.standard_normal((num_slots, HEADS, HEAD_DIM, HEAD_DIM)) * 0.05,
        jnp.float32)
    a_log = jnp.asarray(rng.standard_normal(HEADS) * 0.3, jnp.float32)
    dt_bias = jnp.asarray(rng.standard_normal(PROJECTION) * 0.2, jnp.float32)

    # ``ragged_kda`` takes the raw beta projection and activates it internally;
    # both Pallas kernels want it already through the sigmoid, which the
    # production op does exactly this way -- in float32, rounded back to bf16.
    beta_raw = jnp.asarray(rng.standard_normal((bucket, HEADS)), jnp.bfloat16)
    beta = jnp.asarray(jax.nn.sigmoid(beta_raw.astype(jnp.float32)),
                       jnp.bfloat16)

    return {
        "mixed_qkv": activation(MIXED_DIM),
        "raw_gate": activation(PROJECTION),
        "beta_raw": beta_raw,
        "beta": beta,
        "output_gate": activation(PROJECTION),
        "conv_state": conv_state,
        "conv_weight": conv_weight,
        "pool": pool,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "query_start_loc": jnp.asarray(cumulative, jnp.int32),
        "state_indices": jnp.asarray(slots, jnp.int32),
        "seq_lens": jnp.asarray(lengths, jnp.int32),
        "num_live": num_live,
        "bucket": bucket,
    }


def _call(inputs, *, lower_bound=None, **kwargs):
    """Invoke the kernel the way a custom op would."""
    query_lens = inputs["query_start_loc"][1:] - inputs["query_start_loc"][:-1]
    has_initial_state = inputs["seq_lens"] > query_lens
    return decode_kda(
        inputs["mixed_qkv"],
        inputs["raw_gate"].reshape(inputs["bucket"], HEADS, HEAD_DIM),
        inputs["beta"],
        inputs["conv_state"],
        inputs["conv_weight"],
        inputs["pool"],
        inputs["a_log"],
        # The decode kernel wants dt_bias per head and channel; chunk_kda takes
        # the same numbers flat.
        inputs["dt_bias"].reshape(HEADS, HEAD_DIM),
        inputs["state_indices"],
        has_initial_state,
        lower_bound=lower_bound,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _l2_normalize(x):
    """Row-wise, with the kernel's epsilon."""
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + 1e-6)


def _reference_step(inputs, seq, *, lower_bound):
    """Convolution, SiLU and one KDA step for one sequence, in float64.

    Returns ``(output, new_conv_window, new_state)`` for that sequence.
    """
    slot = int(inputs["state_indices"][seq])
    query_len = int(inputs["query_start_loc"][seq + 1] -
                    inputs["query_start_loc"][seq])
    carries = int(inputs["seq_lens"][seq]) > query_len

    token = np.asarray(inputs["mixed_qkv"][seq]).astype(np.float64)
    conv_state = np.asarray(inputs["conv_state"]).astype(np.float64)
    prior = conv_state[slot]
    if prior.shape[0] == MIXED_DIM:  # conv_state_dim_first
        prior = prior.T
    # A first step starts from zero: the slot may hold another request's tail.
    prior = prior if carries else np.zeros_like(prior)

    weights = np.asarray(inputs["conv_weight"][:, 0, :]).astype(np.float64).T
    window = np.concatenate([prior, token[None]], axis=0)  # [K, mixed_dim]
    # The kernel rounds back to bfloat16 after the activation, so the reference
    # has to as well or the comparison measures the cast.
    conv_out = np.asarray(
        jnp.asarray(_silu(np.sum(window * weights, axis=0)),
                    jnp.bfloat16), ).astype(np.float64)

    q, k, v = (conv_out.reshape(3, HEADS, HEAD_DIM)[i] for i in range(3))
    q = _l2_normalize(q) * SCALE
    k = _l2_normalize(k)

    gate = np.asarray(inputs["raw_gate"][seq]).astype(np.float64).reshape(
        HEADS, HEAD_DIM)
    gate = gate + np.asarray(inputs["dt_bias"]).astype(np.float64).reshape(
        HEADS, HEAD_DIM)
    decay_coeff = np.exp(np.asarray(inputs["a_log"]).astype(np.float64))[:,
                                                                         None]
    if lower_bound is None:
        gate = -decay_coeff * np.logaddexp(0.0, gate)
    else:
        gate = lower_bound * np.exp(-np.logaddexp(0.0, -(decay_coeff * gate)))
    decay = np.exp(gate)  # [H, K]

    state = np.asarray(inputs["pool"][slot]).astype(np.float64)
    state = state if carries else np.zeros_like(state)
    beta = np.asarray(inputs["beta"][seq]).astype(np.float64)

    state = state * decay[:, :, None]
    predicted = np.einsum("hk,hkv->hv", k, state)
    v_new = beta[:, None] * (v - predicted)
    state = state + k[:, :, None] * v_new[:, None, :]
    output = np.einsum("hk,hkv->hv", q, state)
    return output, window[-(KERNEL_SIZE - 1):], state


def _assert_close(actual, expected, *, rtol, name):
    """Compare with an absolute floor tied to the tensor's own scale."""
    expected = np.asarray(expected, np.float64)
    magnitude = max(float(np.abs(expected).max()), 1e-6)
    np.testing.assert_allclose(
        np.asarray(actual, np.float64),
        expected,
        rtol=rtol,
        atol=rtol * magnitude,
        err_msg=f"{name} disagrees with the float64 reference",
    )


# ---------------------------------------------------------------------------
# ABI
# ---------------------------------------------------------------------------


def test_decode_kda_abi_shapes_and_dtypes() -> None:
    """The three returns a caller unpacks, without compiling."""
    inputs = _build(4, 8, seed=3)
    out, conv_state, pool = jax.eval_shape(functools.partial(_call, inputs))

    assert out.shape == (8, HEADS, HEAD_DIM)
    assert out.dtype == jnp.bfloat16
    # Both pools come back in the caller's layout and dtype, ready to be copied
    # straight back into the cache buffers.
    assert conv_state.shape == inputs["conv_state"].shape
    assert conv_state.dtype == inputs["conv_state"].dtype
    assert pool.shape == inputs["pool"].shape
    assert pool.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Correctness against the float64 reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("num_live", "bucket"), [
    (1, 1),
    (1, 8),
    (4, 8),
    (8, 8),
    (3, 16),
])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_decode_kda_matches_the_reference(num_live: int, bucket: int,
                                          lower_bound: float | None) -> None:
    """Output and both state pools, over live/padded splits and both gates."""
    inputs = _build(num_live, bucket, seed=num_live + bucket)
    out, conv_state, pool = _call(inputs, lower_bound=lower_bound)

    out = np.asarray(out, np.float64)
    pool = np.asarray(pool, np.float64)
    conv_state = np.asarray(conv_state, np.float64)

    for seq in range(num_live):
        slot = int(inputs["state_indices"][seq])
        want_out, want_conv, want_state = _reference_step(
            inputs, seq, lower_bound=lower_bound)
        _assert_close(out[seq], want_out, rtol=3e-2, name=f"output[{seq}]")
        _assert_close(pool[slot],
                      want_state,
                      rtol=2e-2,
                      name=f"state[slot {slot}]")
        _assert_close(conv_state[slot],
                      want_conv,
                      rtol=3e-2,
                      name=f"conv state[slot {slot}]")


def test_decode_kda_starts_from_zero_on_a_sequences_first_step() -> None:
    """No carried state means zero, even though the slot holds something else.

    A freshly allocated slot holds whatever the previous occupant left. Reading
    it on a first step would leak one request's context into another, and the
    output would still look plausible.
    """
    inputs = _build(4, 8, seed=11, carry=False)
    # Make the leak impossible to miss if it happens.
    inputs["pool"] = inputs["pool"] * 1000.0
    out, _, pool = _call(inputs)

    for seq in range(4):
        slot = int(inputs["state_indices"][seq])
        want_out, _, want_state = _reference_step(inputs,
                                                  seq,
                                                  lower_bound=None)
        _assert_close(out[seq], want_out, rtol=3e-2, name=f"output[{seq}]")
        _assert_close(np.asarray(pool, np.float64)[slot],
                      want_state,
                      rtol=2e-2,
                      name=f"state[slot {slot}]")


def test_decode_kda_mixes_carried_and_fresh_sequences_in_one_batch() -> None:
    """Carry-in is per sequence, and a real batch is mixed.

    A request on its first decode step sits next to requests mid-stream, so the
    flag has to be read per grid step. A kernel that took it batch-wide -- or
    read the wrong row -- would still pass the all-carried and all-fresh tests.
    """
    inputs = _build(4, 8, seed=41)
    # Sequences 1 and 3 are on their first step; 0 and 2 have context behind
    # them. Every query length is 1, so `seq_lens > 1` is the carry flag.
    inputs["seq_lens"] = jnp.asarray([129, 1, 129, 1] + [0] * 4, jnp.int32)
    inputs["pool"] = inputs["pool"] * 1000.0  # make a wrong read obvious

    out, _, pool = _call(inputs)
    pool = np.asarray(pool, np.float64)
    for seq in range(4):
        slot = int(inputs["state_indices"][seq])
        want_out, _, want_state = _reference_step(inputs,
                                                  seq,
                                                  lower_bound=None)
        _assert_close(out[seq], want_out, rtol=3e-2, name=f"output[{seq}]")
        _assert_close(pool[slot],
                      want_state,
                      rtol=2e-2,
                      name=f"state[slot {slot}]")


# ---------------------------------------------------------------------------
# Paged-state ownership
# ---------------------------------------------------------------------------


def test_decode_kda_zeroes_padded_request_rows() -> None:
    """Rows past the live requests must emit zeros, not stale values."""
    inputs = _build(3, 16, seed=5)
    out, _, _ = _call(inputs)
    np.testing.assert_array_equal(
        np.asarray(out[3:], np.float32),
        np.zeros((13, HEADS, HEAD_DIM), np.float32),
    )


def test_decode_kda_leaves_the_null_block_untouched() -> None:
    """Slot 0 is reserved, and every padded row's write is routed to it.

    If those writes landed anywhere else they would corrupt a live request's
    cache; if they landed on slot 0 *and changed it*, the next step's
    zero-initialised reads would be wrong.
    """
    inputs = _build(3, 16, seed=7)
    _, conv_state, pool = _call(inputs)
    np.testing.assert_array_equal(np.asarray(pool[0]),
                                  np.asarray(inputs["pool"][0]))
    np.testing.assert_array_equal(np.asarray(conv_state[0]),
                                  np.asarray(inputs["conv_state"][0]))


def test_decode_kda_leaves_slots_of_unscheduled_requests_untouched() -> None:
    """A slot that belongs to no row this step must survive the call.

    The runner keeps a request's slot across steps it is not scheduled in, so
    anything written there would be lost context.
    """
    inputs = _build(3, 16, seed=9)
    _, conv_state, pool = _call(inputs)
    # Slots 1..3 are live; 4..16 belong to requests not scheduled this step.
    np.testing.assert_array_equal(np.asarray(pool[4:]),
                                  np.asarray(inputs["pool"][4:]))
    np.testing.assert_array_equal(np.asarray(conv_state[4:]),
                                  np.asarray(inputs["conv_state"][4:]))


def test_decode_kda_advances_only_the_scheduled_slots() -> None:
    """The complement of the checks above: live slots really do change."""
    inputs = _build(3, 16, seed=13)
    _, conv_state, pool = _call(inputs)
    for slot in (1, 2, 3):
        assert not np.allclose(np.asarray(pool[slot]),
                               np.asarray(inputs["pool"][slot]))
        assert not np.allclose(
            np.asarray(conv_state[slot], np.float32),
            np.asarray(inputs["conv_state"][slot], np.float32))


def test_decode_kda_honours_out_of_order_slot_assignments() -> None:
    """Slots are handed out by the block manager, so they are not sorted.

    The grid steps through requests in order while the state specs index by
    slot; mixing the two up would only show when the two orders disagree.
    """
    inputs = _build(4, 8, seed=17)
    inputs["state_indices"] = jnp.asarray([6, 2, 8, 4, 0, 0, 0, 0], jnp.int32)
    out, _, pool = _call(inputs)

    pool = np.asarray(pool, np.float64)
    for seq in range(4):
        slot = int(inputs["state_indices"][seq])
        want_out, _, want_state = _reference_step(inputs,
                                                  seq,
                                                  lower_bound=None)
        _assert_close(out[seq], want_out, rtol=3e-2, name=f"output[{seq}]")
        _assert_close(pool[slot],
                      want_state,
                      rtol=2e-2,
                      name=f"state[slot {slot}]")


# ---------------------------------------------------------------------------
# Layout options
# ---------------------------------------------------------------------------


def test_decode_kda_reads_a_transposed_state_pool() -> None:
    """``state_transposed`` describes a pool holding ``[d_v, d_k]`` per head.

    Since ``d_v == d_k`` the shapes cannot tell the two apart, so a mismatch
    here is invisible until the numbers come out wrong.
    """
    inputs = _build(4, 8, seed=19)
    plain_out, _, plain_pool = _call(inputs)

    swapped = dict(inputs)
    swapped["pool"] = jnp.swapaxes(inputs["pool"], -1, -2)
    out, _, pool = _call(swapped, state_transposed=True)

    _assert_close(out,
                  np.asarray(plain_out, np.float64),
                  rtol=0,
                  name="output")
    _assert_close(np.swapaxes(np.asarray(pool, np.float64), -1, -2),
                  np.asarray(plain_pool, np.float64),
                  rtol=0,
                  name="state")


def test_decode_kda_reads_a_dim_first_conv_state() -> None:
    """``conv_state_dim_first`` is the ``DS`` cache layout, and round-trips."""
    inputs = _build(4, 8, seed=23)
    plain_out, plain_conv, _ = _call(inputs)

    swapped = dict(inputs)
    swapped["conv_state"] = jnp.swapaxes(inputs["conv_state"], 1, 2)
    out, conv_state, _ = _call(swapped, conv_state_dim_first=True)

    assert conv_state.shape == swapped["conv_state"].shape
    _assert_close(out,
                  np.asarray(plain_out, np.float64),
                  rtol=0,
                  name="output")
    np.testing.assert_array_equal(
        np.asarray(jnp.swapaxes(conv_state, 1, 2), np.float32),
        np.asarray(plain_conv, np.float32),
    )


# ---------------------------------------------------------------------------
# Agreement with the path it replaces
# ---------------------------------------------------------------------------


def test_decode_kda_matches_mains_two_op_path() -> None:
    """The fused kernel must be a drop-in for sconv + SiLU + ragged_kda.

    This is the claim the change actually makes, so it is worth pinning directly
    rather than inferring it from two separate reference tests. The tolerance is
    looser than the float64 comparisons because the two paths round differently:
    main casts the convolution to bfloat16 before the SiLU, the kernel after, and
    main activates beta in float32 while the kernel is handed it in bfloat16.
    """
    inputs = _build(6, 16, seed=29)
    norm_weight = jnp.ones((HEAD_DIM, ), jnp.float32)
    fused_out, fused_conv, fused_pool = _call(inputs)

    conv_out, naive_conv = kimi_short_conv_scan(
        inputs["mixed_qkv"],
        inputs["conv_state"],
        inputs["conv_weight"],
        inputs["query_start_loc"],
        inputs["state_indices"],
        inputs["seq_lens"],
    )
    naive_out, naive_pool = ragged_kda(
        jax.nn.silu(conv_out),
        inputs["raw_gate"],
        inputs["beta_raw"],
        inputs["output_gate"],
        inputs["pool"],
        inputs["a_log"],
        inputs["dt_bias"],
        norm_weight,
        inputs["query_start_loc"],
        inputs["state_indices"],
        inputs["seq_lens"],
        lower_bound=None,
        eps=EPS,
    )

    # ``ragged_kda`` folds the output norm and gate in, so apply them to the
    # fused output to land on the same quantity.
    normed = fused_out.astype(jnp.float32)
    normed *= jax.lax.rsqrt(
        jnp.mean(normed * normed, axis=-1, keepdims=True) + EPS)
    normed = (normed * norm_weight).astype(jnp.bfloat16)
    normed = normed * jax.nn.sigmoid(inputs["output_gate"].reshape(
        normed.shape))

    _assert_close(normed,
                  np.asarray(naive_out, np.float64),
                  rtol=4e-2,
                  name="gated output")
    _assert_close(fused_pool,
                  np.asarray(naive_pool, np.float64),
                  rtol=3e-2,
                  name="recurrent pool")
    np.testing.assert_allclose(
        np.asarray(fused_conv, np.float32),
        np.asarray(naive_conv, np.float32),
        rtol=2e-2,
        atol=2e-2,
        err_msg="convolution state disagrees with main's scan",
    )


# ---------------------------------------------------------------------------
# Rejected inputs
# ---------------------------------------------------------------------------


def test_decode_kda_rejects_a_mixed_qkv_width_that_is_not_three_projections(
) -> None:
    inputs = _build(2, 4, seed=31)
    inputs["mixed_qkv"] = inputs["mixed_qkv"][:, :MIXED_DIM - HEAD_DIM]
    with pytest.raises(ValueError, match="mixed_qkv must have width"):
        _call(inputs)


def test_decode_kda_rejects_a_misshaped_conv_weight() -> None:
    inputs = _build(2, 4, seed=33)
    inputs["conv_weight"] = inputs["conv_weight"][:, 0, :]
    with pytest.raises(ValueError, match="conv_weight must have shape"):
        _call(inputs)


def test_decode_kda_rejects_a_conv_state_with_the_wrong_tail() -> None:
    """The tail encodes the layout, so a mismatch means a misread cache."""
    inputs = _build(2, 4, seed=35)
    with pytest.raises(ValueError, match="conv_state must end in"):
        _call(inputs, conv_state_dim_first=True)


def test_decode_kda_rejects_a_dt_bias_in_the_chunk_kernels_layout() -> None:
    """``chunk_kda`` takes dt_bias flat; this kernel wants it per head.

    Passing the flat form would otherwise be a silent broadcast.
    """
    inputs = _build(2, 4, seed=37)
    query_lens = inputs["query_start_loc"][1:] - inputs["query_start_loc"][:-1]
    with pytest.raises(ValueError, match="dt_bias must be"):
        decode_kda(
            inputs["mixed_qkv"],
            inputs["raw_gate"].reshape(inputs["bucket"], HEADS, HEAD_DIM),
            inputs["beta"],
            inputs["conv_state"],
            inputs["conv_weight"],
            inputs["pool"],
            inputs["a_log"],
            inputs["dt_bias"],  # flat [H * D], not [H, D]
            inputs["state_indices"],
            inputs["seq_lens"] > query_lens,
        )


def test_decode_kda_rejects_unequal_key_and_value_dims() -> None:
    inputs = _build(2, 4, seed=39)
    inputs["pool"] = inputs["pool"][..., :HEAD_DIM // 2]
    with pytest.raises(ValueError, match="d_v == d_k"):
        _call(inputs)
