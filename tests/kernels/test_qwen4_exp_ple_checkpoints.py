# SPDX-License-Identifier: Apache-2.0
"""Token-major PLE checkpoints write exactly the request-major (target, value) pairs.

Pure JAX (no Pallas), so it runs on CPU: the writer kernel is shared, so equal
non-null pairs mean equal pool bytes.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv
from vllm_torchtpu.kernels.qwen4_exp.pool import (
    request_checkpoints,
    token_checkpoints,
)

STATE_LEN, WINDOW, CHANNELS = 9, 4, 128


def _pairs(values, targets):
    values, targets = np.asarray(values), np.asarray(targets)
    out = {}
    for row, target in enumerate(targets):
        if target > 0:
            assert target not in out, f"duplicate target {target}"
            out[int(target)] = values[row].view(np.uint16)
    return out


def _case(rng, lengths, windowed, ids_on, initial, tokens):
    count = len(lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    assert starts[-1] <= tokens
    ids = np.where(ids_on, 1 + np.arange(count), 0).astype(np.int32)
    ckpt = (1 + np.arange(count * WINDOW, dtype=np.int32)).reshape(count, WINDOW)
    x = jnp.asarray(rng.normal(size=(tokens, CHANNELS)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(count, STATE_LEN, CHANNELS)), jnp.bfloat16)
    starts, initial = jnp.asarray(starts), jnp.asarray(initial)
    lengths = starts[1:] - starts[:-1]
    active = (jnp.asarray(ids) > 0) & (lengths > 0)
    slots = jnp.where(active, jnp.arange(count, dtype=jnp.int32), -1)
    weight = jnp.asarray(rng.normal(size=(CHANNELS, 4)), jnp.bfloat16)
    _, final = short_conv(
        x,
        weight,
        history,
        starts,
        slots,
        initial,
        dilation=3,
        request_owned_history=True,
    )
    args = (
        x,
        history,
        starts,
        initial,
        active,
        jnp.asarray(ckpt),
        jnp.asarray(windowed),
    )
    old = _pairs(*request_checkpoints(x, history, final, *args[2:]))
    new = _pairs(*token_checkpoints(*args))
    return old, new


@pytest.mark.parametrize(
    "lengths,windowed,ids_on,initial,tokens",
    [
        # verify windows, one idle row, one short row, padded tail
        ([4, 4, 0, 2, 3], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 1, 0], 16),
        # decode rows (one token each)
        ([1, 1, 1, 1], [1] * 4, [1, 0, 1, 1], [1, 1, 0, 1], 4),
        # leading windows then a prefill row with and without history
        ([4, 3, 6, 5], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 1], 18),
        # a prefill row shorter than the state, idle prefill, zero-length windowed
        ([2, 0, 3, 7], [0, 1, 0, 0], [1, 1, 0, 1], [0, 1, 1, 1], 12),
    ],
)
def test_token_checkpoints_match_request_checkpoints(
    lengths, windowed, ids_on, initial, tokens
):
    old, new = _case(
        np.random.default_rng(len(lengths) * 7 + tokens),
        lengths,
        np.asarray(windowed, bool),
        np.asarray(ids_on, bool),
        np.asarray(initial, bool),
        tokens,
    )
    assert old, "case writes nothing"
    assert old.keys() == new.keys()
    for target in old:
        np.testing.assert_array_equal(
            new[target], old[target], err_msg=f"target {target}"
        )


def test_token_checkpoints_catch_an_off_by_one_window():
    """Positive control: shifting the token windows by one must be detected."""
    rng = np.random.default_rng(3)
    old, new = _case(
        rng,
        [4, 4],
        np.array([1, 1], bool),
        np.array([1, 1], bool),
        np.array([1, 1], bool),
        8,
    )
    shifted = {k: new[k + 1] for k in new if k + 1 in new}
    assert any(not np.array_equal(shifted[k], old[k]) for k in shifted)


def _token_checkpoints_pair_gather(x, history, starts, initial, active, ckpt_ids, windowed):
    """`token_checkpoints` before the flat-row history gather: the history indexed by a
    (request, position) pair of index arrays. Kept as the bitwise reference."""
    token_count = x.shape[0]
    count, window = ckpt_ids.shape
    state_len = history.shape[1]
    lengths = starts[1:] - starts[:-1]
    t = jnp.arange(token_count, dtype=jnp.int32)
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), count - 1)
    rel = t - starts[req]
    positions = (
        rel[:, None] + 1 - state_len + jnp.arange(state_len, dtype=jnp.int32)[None]
    )
    past = history[req[:, None], jnp.clip(positions + state_len, 0, state_len - 1)]
    past = jnp.where(initial[req][:, None, None], past, 0)
    current = x[jnp.clip(starts[req][:, None] + positions, 0, max(token_count - 1, 0))]
    values = jnp.where((positions >= 0)[..., None], current, past.astype(x.dtype))
    win = windowed[req]
    write = (
        (t < starts[-1])
        & active[req]
        & jnp.where(win, rel < window, rel == lengths[req] - 1)
    )
    column = jnp.where(win, jnp.clip(rel, 0, window - 1), 0)
    targets = jnp.where(write, ckpt_ids[req, column], 0)
    return values.reshape(token_count, -1), targets


def _served_starts(rng, count, tokens, kind):
    if kind == "verify":  # every row a window of up to WINDOW tokens, padded rows repeat
        lengths = rng.integers(0, WINDOW + 1, size=count)
    elif kind == "decode":
        lengths = np.ones(count, np.int64)
    else:  # mixed: windows plus prefill rows, possibly past the token count
        lengths = rng.integers(0, max(2, tokens // 2), size=count)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    return np.minimum(starts, tokens + 8).astype(np.int32)


@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("kind", ["verify", "decode", "mixed"])
@pytest.mark.parametrize("channels", [128, 10240])
def test_flat_history_gather_is_bitwise_the_pair_gather(tokens, kind, channels):
    """Served PLE shapes (16 request rows, window 4, state 9): the flat-row gather
    returns the old values and targets bit for bit, including the rows of
    inactive / non-initial requests and starts past the token count."""
    rng = np.random.default_rng(tokens * 31 + len(kind) + channels)
    count = 16
    starts = jnp.asarray(_served_starts(rng, count, tokens, kind))
    x = jnp.asarray(rng.normal(size=(tokens, channels)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(count, STATE_LEN, channels)), jnp.bfloat16)
    initial = jnp.asarray(rng.integers(0, 2, size=count).astype(bool))
    active = jnp.asarray(rng.integers(0, 4, size=count) > 0)
    ckpt = jnp.asarray((1 + np.arange(count * WINDOW, dtype=np.int32)).reshape(count, WINDOW))
    windowed = jnp.asarray(rng.integers(0, 4, size=count) > 0)
    args = (x, history, starts, initial, active, ckpt, windowed)
    ref_v, ref_t = _token_checkpoints_pair_gather(*args)
    got_v, got_t = token_checkpoints(*args)
    assert got_v.dtype == ref_v.dtype and got_v.shape == ref_v.shape
    np.testing.assert_array_equal(
        np.asarray(got_v).view(np.uint16), np.asarray(ref_v).view(np.uint16)
    )
    np.testing.assert_array_equal(np.asarray(got_t), np.asarray(ref_t))


def test_flat_history_gather_catches_a_wrong_request_row():
    """Positive control: reading the neighbouring request's history must be detected."""
    rng = np.random.default_rng(5)
    count, tokens = 16, 16
    starts = jnp.asarray(_served_starts(rng, count, tokens, "verify"))
    x = jnp.asarray(rng.normal(size=(tokens, CHANNELS)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(count, STATE_LEN, CHANNELS)), jnp.bfloat16)
    args = (
        x,
        history,
        starts,
        jnp.ones(count, bool),
        jnp.ones(count, bool),
        jnp.asarray((1 + np.arange(count * WINDOW, dtype=np.int32)).reshape(count, WINDOW)),
        jnp.ones(count, bool),
    )
    ref_v, _ = _token_checkpoints_pair_gather(*args)
    got_v, _ = token_checkpoints(x, jnp.roll(history, 1, axis=0), *args[2:])
    assert not np.array_equal(np.asarray(got_v).view(np.uint16), np.asarray(ref_v).view(np.uint16))
