# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_CONV_ONEHOT (P-DENSE conv1h): short_conv(past_matmul=True) reads the history taps by an exact
one-hot matmul; bitwise equal to the gather. CPU (jit), no Pallas."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv

STREAMS, HIDDEN, TAPS, DIL, EPS, SLOTS = 4, 128, 4, 3, 1e-6, 8
WIDTH = STREAMS * HIDDEN
STATE = (TAPS - 1) * DIL


def _layout(name, tokens):
    if name == "one":
        lengths, ids, init = [tokens], [5], [True]
    elif name == "two":
        a, valid = tokens * 5 // 16, tokens - tokens // 16
        lengths, ids, init = [a, valid - a], [5, 9], [True, False]
    elif name == "mix":
        valid = tokens - 37
        lengths = [5, 3, 11, 1, 7, tokens // 4, 2]
        lengths.append(valid - sum(lengths))
        ids, init = list(range(1, 9)), [True, False, True, True, False, True, True, False]
    elif name == "idle":  # an active request between an id-0 slot and padding
        lengths, ids, init = [40, tokens - 40 - 16], [0, 7], [True, True]
    else:
        raise ValueError(name)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    n = len(lengths)
    valid = starts[n]
    starts += [starts[-1]] * (SLOTS - n)
    ids += [-1] * (SLOTS - n)
    init += [False] * (SLOTS - n)
    return (
        jnp.asarray(starts, jnp.int32),
        jnp.asarray(ids, jnp.int32),
        jnp.asarray(init, bool),
        jnp.asarray(valid, jnp.int32),
    )


def _bf16(rng, shape, scale=1.0):
    return jnp.asarray((rng.normal(size=shape) * scale).astype(np.float32), jnp.bfloat16)


def _owner_mask(starts, ids, valid, tokens):
    """forward_with_history's mask with token_requests' cumsum owner, in numpy."""
    starts, ids = np.asarray(starts), np.asarray(ids)
    marks = np.zeros(tokens + 1, np.int64)
    for s in np.minimum(starts[1:SLOTS], tokens):
        marks[s] += 1
    owner = np.minimum(np.cumsum(marks[:tokens]), SLOTS - 1)
    return (np.arange(tokens) < int(valid)) & (ids[owner] >= 0)


@pytest.mark.parametrize("layout", ["one", "two", "mix", "idle"])
def test_onehot_history_is_bitwise(layout):
    tokens = 256
    rng = np.random.default_rng(3)
    x = _bf16(rng, (tokens, WIDTH))
    weight = _bf16(rng, (WIDTH, TAPS), 0.5)
    history = np.array(_bf16(rng, (SLOTS, STATE, WIDTH)))
    starts, ids, init, _ = _layout(layout, tokens)
    # A non-finite row in a slot without initial state must not reach any output.
    unused = int(np.flatnonzero(~np.asarray(init))[0])
    history[unused, 2, 7] = np.nan
    history = jnp.asarray(history)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(SLOTS, dtype=jnp.int32), -1)
    run = jax.jit(short_conv, static_argnames=("dilation", "request_owned_history", "past_matmul"))
    ref = run(x, weight, history, starts, slots, init, dilation=DIL, request_owned_history=True)
    got = run(
        x, weight, history, starts, slots, init, dilation=DIL,
        request_owned_history=True, past_matmul=True,
    )
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16)
        )


@pytest.mark.parametrize("layout", ["one", "two", "mix"])
def test_onehot_probe_mutant_changes_history_rows(layout):
    """Q38_PROBE_PLE_ONEHOT_MUTANT's short_conv(past_matmul="mutant") must differ from the gather wherever history
    is read (the identity probe's must-differ control), and only there."""
    tokens = 256
    rng = np.random.default_rng(4)
    x, weight = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (WIDTH, TAPS), 0.5)
    history = _bf16(rng, (SLOTS, STATE, WIDTH))
    starts, ids, init, _ = _layout(layout, tokens)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(SLOTS, dtype=jnp.int32), -1)
    ref, _ = short_conv(x, weight, history, starts, slots, init, dilation=DIL, request_owned_history=True)
    mut, _ = short_conv(x, weight, history, starts, slots, init, dilation=DIL, request_owned_history=True,
                        past_matmul="mutant")
    differ = np.any(np.asarray(ref).view(np.uint16) != np.asarray(mut).view(np.uint16), axis=1)
    s, init_np = np.asarray(starts), np.asarray(init)
    hist_rows = np.zeros(tokens, bool)
    for r in range(SLOTS):
        if init_np[r] and np.asarray(active)[r]:
            hist_rows[s[r]:min(s[r] + STATE, s[r + 1])] = True
    assert differ.any() and not differ[~hist_rows].any()
