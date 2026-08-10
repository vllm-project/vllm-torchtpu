# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the chunked Kimi Delta Attention Pallas kernel."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.kimi_k3 import chunk_kda

HEADS = 2
HEAD_DIM = 128
CHUNK = 64
SCALE = HEAD_DIM**-0.5


def _random_activations(num_tokens: int, seed: int) -> dict[str, jax.Array]:
    """Head-major ``[H, B=1, T, D]`` bfloat16 activations, as the kernel wants."""
    rng = np.random.default_rng(seed)

    def normal(last_dim):
        return jnp.asarray(
            rng.standard_normal((HEADS, 1, num_tokens, last_dim)),
            jnp.bfloat16)

    raw_beta = rng.standard_normal((HEADS, 1, num_tokens))
    return {
        "q": normal(HEAD_DIM),
        "k": normal(HEAD_DIM),
        "v": normal(HEAD_DIM),
        "g": normal(HEAD_DIM),
        # The Pallas kernels take beta already through the sigmoid; the custom
        # op does that on the way in.
        "beta": jnp.asarray(np.exp(-np.logaddexp(0.0, -raw_beta)),
                            jnp.bfloat16),
    }


def _l2_normalize(x: jax.Array) -> jax.Array:
    """Mirror ``_preprocess_inputs``: normalise in float32, store back in bf16."""
    x_f32 = np.asarray(x).astype(np.float32)
    inv_norm = 1.0 / np.sqrt(
        np.sum(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6)
    return jnp.asarray(x_f32 * inv_norm, x.dtype)


def _activate_gate(
    g: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    lower_bound: float | None,
) -> np.ndarray:
    """Mirror the in-kernel gate activation, in float32 as the kernel does.

    Returns the natural-log decay per token and channel. The kernel keeps it in
    base 2 (``cumsum_scale = 1 / ln 2`` and ``exp2``); that is the same number.
    """
    gate = np.asarray(g).astype(np.float32)
    gate = gate + np.asarray(dt_bias).reshape(HEADS, 1, 1, HEAD_DIM)
    decay = np.exp(np.asarray(a_log)).reshape(HEADS, 1, 1, 1)
    if lower_bound is None:
        return -decay * np.logaddexp(0.0, gate)
    return lower_bound * np.exp(-np.logaddexp(0.0, -(decay * gate)))


def _recurrence(
    q: np.ndarray,  # [H, T, K], already normalised
    k: np.ndarray,  # [H, T, K], already normalised
    v: np.ndarray,  # [H, T, V]
    gate: np.ndarray,  # [H, T, K], already activated (natural log)
    beta: np.ndarray,  # [H, T], already through the sigmoid
    state: np.ndarray,  # [H, K, V]
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One token at a time, in float64. The gold standard for every test here."""
    q, k, v = (np.asarray(x).astype(np.float64) for x in (q, k, v))
    gate = np.asarray(gate).astype(np.float64)
    beta = np.asarray(beta).astype(np.float64)
    running = np.asarray(state).astype(np.float64).copy()

    output = np.zeros((q.shape[0], q.shape[1], v.shape[-1]), np.float64)
    for token in range(q.shape[1]):
        running *= np.exp(gate[:, token])[:, :, None]
        prediction = np.einsum("hk,hkv->hv", k[:, token], running)
        delta = beta[:, token][:, None] * (v[:, token] - prediction)
        running += k[:, token][:, :, None] * delta[:, None, :]
        output[:,
               token] = scale * np.einsum("hk,hkv->hv", q[:, token], running)
    return output, running


def _assert_close(actual, expected, *, rtol: float, name: str) -> None:
    """Compare with an absolute floor tied to the tensor's own scale.

    The kernel returns bfloat16 activations, so the smallest entries of a tensor
    carry no information and a pure ``rtol`` would test rounding noise. Tying
    ``atol`` to ``max|expected|`` keeps the check meaningful on the entries that
    matter.
    """
    expected = np.asarray(expected, np.float64)
    magnitude = max(float(np.abs(expected).max()), 1e-6)
    np.testing.assert_allclose(
        np.asarray(actual, np.float64),
        expected,
        rtol=rtol,
        atol=rtol * magnitude,
        err_msg=f"{name} disagrees with the token-at-a-time recurrence",
    )


def _segment_ids(query_lens, num_tokens: int) -> jax.Array:
    """1-indexed segment ids over the token axis, 0 for the padded tail."""
    ids = np.zeros((1, num_tokens), np.int32)
    start = 0
    for segment, length in enumerate(query_lens, start=1):
        ids[0, start:start + length] = segment
        start += length
    return jnp.asarray(ids)


def _run(
    query_lens,
    *,
    num_tokens: int | None = None,
    num_segments: int | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    with_initial_state: bool = False,
    a_log_value: float = 0.0,
    start_seq: int | None = None,
    seed: int = 0,
):
    """Run ``chunk_kda`` and the reference on the same inputs.

    Returns ``(output, expected_output, final_state, expected_final_state)``.
    """
    if num_tokens is None:
        num_tokens = sum(query_lens)
    if num_segments is None:
        num_segments = len(query_lens)

    rng = np.random.default_rng(seed + 977)
    activations = _random_activations(num_tokens, seed)
    a_log = jnp.full((HEADS, ), a_log_value, jnp.float32)
    dt_bias = jnp.asarray(rng.standard_normal(HEADS * HEAD_DIM), jnp.float32)

    # The reference always needs the activated gate. When the caller asks for
    # the pre-activated path, the kernel sees it rounded to bfloat16, so the
    # reference has to see the same rounded values.
    gate = _activate_gate(activations["g"], a_log, dt_bias, lower_bound)
    if use_gate_in_kernel:
        gate_in = activations["g"]
    else:
        gate_in = jnp.asarray(gate, jnp.bfloat16)
        gate = np.asarray(gate_in).astype(np.float32)

    normalized = {
        name: _l2_normalize(activations[name])
        for name in ("q", "k")
    }
    if use_qk_l2norm_in_kernel:
        q_in, k_in = activations["q"], activations["k"]
    else:
        q_in, k_in = normalized["q"], normalized["k"]

    if with_initial_state:
        state = jnp.asarray(
            rng.standard_normal(
                (1, num_segments, HEADS, HEAD_DIM, HEAD_DIM)) * 0.05,
            jnp.float32)
        state_in = state
    else:
        state = jnp.zeros((1, num_segments, HEADS, HEAD_DIM, HEAD_DIM),
                          jnp.float32)
        state_in = None

    output, final_state = chunk_kda(
        q_in,
        k_in,
        activations["v"],
        gate_in,
        activations["beta"],
        A_log=a_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        scale=SCALE,
        initial_state=state_in,
        output_final_state=True,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=_segment_ids(query_lens, num_tokens),
        lower_bound=lower_bound,
        chunk_size=CHUNK,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        N_max=num_segments,
        start_seq=start_seq,
    )

    expected_output = np.zeros((HEADS, num_tokens, HEAD_DIM), np.float64)
    expected_state = np.asarray(state, np.float64).copy()
    start = 0
    for segment, length in enumerate(query_lens):
        span = slice(start, start + length)
        if start_seq is not None and segment < start_seq:
            # Restricted away: no output, and the state stays as it came in.
            start += length
            continue
        segment_output, segment_state = _recurrence(
            normalized["q"][:, 0, span],
            normalized["k"][:, 0, span],
            activations["v"][:, 0, span],
            gate[:, 0, span],
            activations["beta"][:, 0, span],
            expected_state[0, segment],
            SCALE,
        )
        expected_output[:, span] = segment_output
        expected_state[0, segment] = segment_state
        start += length

    return output[:, 0], expected_output, final_state, expected_state


# ---------------------------------------------------------------------------
# ABI
# ---------------------------------------------------------------------------


def test_chunk_kda_abi_shapes_and_dtypes() -> None:
    """Shapes and dtypes of the pair the layer unpacks, without compiling."""
    num_tokens, num_segments = 128, 3
    activations = _random_activations(num_tokens, seed=3)
    state = jnp.zeros((1, num_segments, HEADS, HEAD_DIM, HEAD_DIM),
                      jnp.float32)

    bound = functools.partial(
        chunk_kda,
        A_log=jnp.zeros((HEADS, ), jnp.float32),
        dt_bias=jnp.zeros((HEADS * HEAD_DIM, ), jnp.float32),
        initial_state=state,
        output_final_state=True,
        use_gate_in_kernel=True,
        segment_ids=_segment_ids([64, 33], num_tokens),
        use_qk_l2norm_in_kernel=True,
        N_max=num_segments,
    )
    output, final_state = jax.eval_shape(
        bound,
        activations["q"],
        activations["k"],
        activations["v"],
        activations["g"],
        activations["beta"],
    )

    assert output.shape == (HEADS, 1, num_tokens, HEAD_DIM)
    assert output.dtype == jnp.bfloat16
    # One state per request slot, so the caller can scatter straight into its
    # pool with the slot indices it already has.
    assert final_state.shape == (1, num_segments, HEADS, HEAD_DIM, HEAD_DIM)
    assert final_state.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Correctness against the token-at-a-time recurrence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [64, 128, 256])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_chunk_kda_matches_recurrence_single_sequence(num_tokens: int,
                                                      lower_bound: float
                                                      | None) -> None:
    """One sequence filling the token axis, over one chunk and over several.

    Also covers both gate activations: the unbounded softplus form and the
    bounded sigmoid one.
    """
    output, expected, _, _ = _run([num_tokens], lower_bound=lower_bound)
    _assert_close(output, expected, rtol=3e-2, name="output")


@pytest.mark.parametrize(
    "query_lens",
    [
        [64],  # exactly one chunk
        [70],  # a short trailing chunk
        [1],  # a single token
        [64, 64],  # two aligned sequences
        [130, 64],  # a sequence spanning three chunks, then one
        [17, 48, 96],  # nothing aligned
        [1, 1, 1, 1],  # decode-shaped
    ],
)
def test_chunk_kda_matches_recurrence_varlen(query_lens) -> None:
    """Ragged batches: every sequence must be solved independently.

    This is what the chunk alignment exists for -- a chunk must never mix
    tokens from two sequences, whatever the lengths are.
    """
    output, expected, _, _ = _run(query_lens)
    _assert_close(output, expected, rtol=3e-2, name="output")


@pytest.mark.parametrize("a_log_value", [0.0, -2.3])
def test_chunk_kda_matches_recurrence_with_slow_decay(
        a_log_value: float) -> None:
    """A gate near zero keeps far-back tokens alive, so chunks must compose.

    With a fast decay a chunked kernel can be wrong about the inter-chunk carry
    and still look right, because the carried state has already decayed away.
    ``A_log = -2.3`` makes the per-token decay about 0.9 and removes that cover.
    """
    output, expected, _, _ = _run([200], a_log_value=a_log_value)
    _assert_close(output, expected, rtol=3e-2, name="output")


# ---------------------------------------------------------------------------
# Recurrent state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_lens", [[128], [70, 33], [1, 1]])
def test_chunk_kda_returns_the_final_state(query_lens) -> None:
    """The state handed back must be the one after the sequence's last token."""
    _, _, final_state, expected_state = _run(query_lens)
    _assert_close(final_state, expected_state, rtol=3e-2, name="final state")


@pytest.mark.parametrize("query_lens", [[128], [70, 33]])
def test_chunk_kda_carries_the_initial_state_in(query_lens) -> None:
    """Prefix state must enter the first chunk, not just the first token.

    A sequence that resumes from cached context starts with a non-zero state,
    and both the intra-chunk output and the carried state depend on it.
    """
    output, expected_output, final_state, expected_state = _run(
        query_lens, with_initial_state=True)
    _assert_close(output, expected_output, rtol=3e-2, name="output")
    _assert_close(final_state, expected_state, rtol=3e-2, name="final state")


def test_chunk_kda_leaves_trailing_empty_segments_at_their_initial_state(
) -> None:
    """A padded request slot must come back unchanged, not zeroed.

    The runner sizes the batch axis for the worst case, so most steps hand the
    kernel segments with no tokens. Their rows never reach kernel code, and
    writing anything but the incoming state there would corrupt a live cache
    entry.
    """
    _, _, final_state, expected_state = _run(
        [64, 33],
        num_segments=5,
        with_initial_state=True,
    )
    final_state = np.asarray(final_state, np.float64)
    # Segments 2..4 hold no tokens: bit-for-bit the state they came in with.
    np.testing.assert_array_equal(final_state[0, 2:], expected_state[0, 2:])
    # And the populated ones did advance.
    assert not np.allclose(final_state[0, :2], expected_state[0, :2])


def test_chunk_kda_zeroes_the_padded_token_rows() -> None:
    """Output rows past the last real token must be zero, not stale.

    The token axis is padded up to the runner's bucket. Every tile scatters only
    the rows it owns, so nothing writes this range and it has to arrive zeroed.
    """
    num_tokens = 128
    output, expected, _, _ = _run([40, 24], num_tokens=num_tokens)
    _assert_close(output, expected, rtol=3e-2, name="output")
    np.testing.assert_array_equal(
        np.asarray(output[:, 64:], np.float32),
        np.zeros((HEADS, num_tokens - 64, HEAD_DIM), np.float32),
    )


@pytest.mark.parametrize(
    "query_lens",
    [
        [1] * 8,  # every tile is one row of a CHUNK-row buffer
        [3, 5, 7],  # short tiles back to back
        [1, 130, 1],  # a one-token segment either side of a long one
        [70, 1],  # a short trailing tile followed by one token
    ],
)
def test_chunk_kda_masks_the_rows_a_tile_does_not_own(query_lens) -> None:
    """A tile shorter than a chunk must ignore the rest of its staging buffer.

    The buffer is reused across tiles and only the owned rows are DMA'd into it,
    so the rest still holds the *previous* tile's tokens -- a different segment.
    Those rows are masked to zero. If they were not, a short segment would mix in
    its neighbour's tokens, and the chunk's total gate decay -- which stage 3
    applies to the whole recurrent state -- would over-decay it.
    """
    output, expected_output, final_state, expected_state = _run(
        query_lens, with_initial_state=True)
    _assert_close(output, expected_output, rtol=3e-2, name="output")
    _assert_close(final_state, expected_state, rtol=3e-2, name="final state")


@pytest.mark.parametrize("start_seq", [0, 1, 2, 4])
def test_chunk_kda_restricted_to_segments_from_start_seq(start_seq) -> None:
    """``start_seq`` hands the leading segments to somebody else.

    This is what lets a caller send decode requests to the fused decode kernel
    and the rest here, with the boundary a device scalar rather than a host-side
    branch -- which vLLM's guard-free single trace makes unusable. Restricted
    segments must produce no output and, critically, must leave their recurrent
    state exactly as it arrived: the other kernel has already written it, and
    re-writing a stale copy would silently corrupt a live cache entry.

    ``start_seq=4`` restricts every segment away, which is the pure-decode step.
    """
    query_lens = [1, 1, 70, 33]
    num_tokens = 128
    output, expected_output, final_state, expected_state = _run(
        query_lens,
        num_tokens=num_tokens,
        with_initial_state=True,
        start_seq=start_seq,
    )
    _assert_close(output, expected_output, rtol=3e-2, name="output")
    _assert_close(final_state, expected_state, rtol=3e-2, name="final state")

    skipped_rows = sum(query_lens[:start_seq])
    np.testing.assert_array_equal(
        np.asarray(output[:, :skipped_rows], np.float32),
        np.zeros((HEADS, skipped_rows, HEAD_DIM), np.float32),
    )
    # Bit-for-bit, not merely close: nothing may touch these slots.
    np.testing.assert_array_equal(
        np.asarray(final_state[0, :start_seq]),
        np.asarray(expected_state[0, :start_seq], np.float32),
    )


def test_chunk_kda_start_seq_agrees_with_an_unrestricted_run() -> None:
    """The segments it does keep must be untouched by the restriction.

    Restricting the front of the batch may not perturb the rest -- the tile plan
    shifts, so the kept segments land on different tile indices.
    """
    query_lens = [1, 1, 70, 33]
    kept_from = sum(query_lens[:2])
    full, _, full_state, _ = _run(query_lens,
                                  num_tokens=128,
                                  with_initial_state=True)
    restricted, _, restricted_state, _ = _run(query_lens,
                                              num_tokens=128,
                                              with_initial_state=True,
                                              start_seq=2)
    np.testing.assert_array_equal(np.asarray(restricted[:, kept_from:]),
                                  np.asarray(full[:, kept_from:]))
    np.testing.assert_array_equal(np.asarray(restricted_state[0, 2:]),
                                  np.asarray(full_state[0, 2:]))


def test_chunk_kda_is_unchanged_by_the_size_of_the_request_bucket() -> None:
    """Padding the request axis must not change the answer, bit for bit.

    The runner fixes the request axis at startup, so the same batch is handed to
    the kernel with a differently sized axis depending on configuration. All the
    extra slots do is add tiles past the device-valued ``num_tiles`` bound.

    This pins agreement, not cost. That padded tiles are also *cheap* -- the
    point of the (sequence, tile) grid -- is a timing property, measured by
    ``scripts/kernels/bench_chunk_kda.py`` rather than asserted here.
    """
    tight, expected, tight_state, expected_state = _run(
        [40, 24], num_tokens=128, num_segments=2, with_initial_state=True)
    padded, _, padded_state, _ = _run([40, 24],
                                      num_tokens=128,
                                      num_segments=160,
                                      with_initial_state=True)
    _assert_close(tight, expected, rtol=3e-2, name="output")
    # Same segments, same tiles: the two runs must agree exactly, not merely to
    # within the reference's tolerance.
    np.testing.assert_array_equal(np.asarray(padded), np.asarray(tight))
    np.testing.assert_array_equal(np.asarray(padded_state[0, :2]),
                                  np.asarray(tight_state[0, :2]))
    _assert_close(tight_state, expected_state, rtol=3e-2, name="final state")


# ---------------------------------------------------------------------------
# The two "already done on the host" flags
# ---------------------------------------------------------------------------


def test_chunk_kda_accepts_a_pre_activated_gate() -> None:
    """``use_gate_in_kernel=False``: ``g`` is the log decay, taken as given."""
    output, expected, _, _ = _run([70, 58], use_gate_in_kernel=False)
    _assert_close(output, expected, rtol=3e-2, name="output")


def test_chunk_kda_accepts_pre_normalized_qk() -> None:
    """``use_qk_l2norm_in_kernel=False``: q and k are used as handed over."""
    output, expected, _, _ = _run([70, 58], use_qk_l2norm_in_kernel=False)
    _assert_close(output, expected, rtol=3e-2, name="output")


# ---------------------------------------------------------------------------
# Rejected inputs
# ---------------------------------------------------------------------------


def _valid_kwargs(num_tokens: int = 64) -> dict:
    """A minimal accepted call, so each test below breaks exactly one thing."""
    activations = _random_activations(num_tokens, seed=11)
    return {
        "q": activations["q"],
        "k": activations["k"],
        "v": activations["v"],
        "g": activations["g"],
        "beta": activations["beta"],
        "segment_ids": _segment_ids([num_tokens], num_tokens),
        "N_max": 1,
    }


def test_chunk_kda_rejects_float32_activations() -> None:
    """The fp32 intra-chunk solve was removed; only bfloat16 is supported."""
    args = _valid_kwargs()
    for name in ("q", "k", "v", "g", "beta"):
        args[name] = args[name].astype(jnp.float32)
    with pytest.raises(NotImplementedError, match="bfloat16"):
        chunk_kda(**args)


def test_chunk_kda_rejects_other_chunk_sizes() -> None:
    with pytest.raises(NotImplementedError, match="chunk_size=64"):
        chunk_kda(**_valid_kwargs(), chunk_size=32)


def test_chunk_kda_requires_segment_ids() -> None:
    """There is no fixed-length mode: a ragged batch is the only interface.

    The kernel used to also accept ``segment_ids=None`` and skip the alignment
    gather. No caller used it, and it doubled the number of state ranks and
    segment-boundary conventions every downstream branch had to handle.
    """
    args = _valid_kwargs()
    del args["segment_ids"]
    with pytest.raises(ValueError, match="segment_ids` is required"):
        chunk_kda(**args)


def test_chunk_kda_rejects_a_four_dimensional_initial_state() -> None:
    """State is one per request slot, so the segment axis is not optional.

    ``[B, H, K, V]`` used to be accepted. Axis 1 is now read as the segment
    count, so taking it would quietly mistake heads for requests.
    """
    args = _valid_kwargs()
    state = jnp.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), jnp.float32)
    with pytest.raises(ValueError, match="initial_state` must be"):
        chunk_kda(**args, initial_state=state)


@pytest.mark.parametrize("lower_bound", [0.0, 5.0])
def test_chunk_kda_rejects_a_non_negative_lower_bound(
        lower_bound: float) -> None:
    """A non-negative bound makes the gate grow along a chunk.

    The intra-chunk decay is only numerically valid while the cumulative gate
    decreases; a positive bound would return quietly wrong values, so it has to
    be refused rather than clamped.
    """
    with pytest.raises(ValueError, match="lower_bound must be negative"):
        chunk_kda(**_valid_kwargs(), lower_bound=lower_bound)


@pytest.mark.parametrize(
    ("broken", "axis"),
    [
        ("k", -1),  # k's channel count must equal q's
        ("g", -1),  # so must the gate's
        ("v", 2),  # v may have its own V, but not its own token count
        ("beta", 2),  # beta is one scalar per token per head
    ],
)
def test_chunk_kda_rejects_mismatched_activation_shapes(
        broken: str, axis: int) -> None:
    """q, k and g share ``[H, B, T, K]``; v and beta share ``[H, B, T]``.

    Nothing downstream re-checks this, and a silently broadcast axis would
    produce a plausible-looking wrong answer rather than an error.
    """
    args = _valid_kwargs()
    shape = list(args[broken].shape)
    shape[axis] //= 2
    args[broken] = jnp.zeros(tuple(shape), args[broken].dtype)
    with pytest.raises(ValueError, match="must"):
        chunk_kda(**args)


def test_chunk_kda_requires_n_max_without_an_initial_state() -> None:
    """The segment count is a static shape, so it cannot be inferred."""
    args = _valid_kwargs()
    del args["N_max"]
    with pytest.raises(ValueError, match="N_max"):
        chunk_kda(**args)
