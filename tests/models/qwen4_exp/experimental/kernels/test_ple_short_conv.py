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
"""Request-owned dilated short convolution against a per-request NumPy
reference: outputs, updated history, padded and empty requests, and state
carried across consecutive steps."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.ple import short_conv

CHANNELS = 128
SLOTS = 6


def _bf16(x):
    return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))


def _reference(x, weight, history, starts, slots, initial, dilation):
    """Per request: conv over [history (or zeros); request tokens]."""
    taps = weight.shape[-1]
    state_len = history.shape[1]
    output = np.zeros_like(x)
    history = history.copy()
    updated = np.zeros((len(slots), state_len, x.shape[1]), np.float32)
    for r, slot in enumerate(slots):
        begin, end = starts[r], starts[r + 1]
        past = history[slot] if slot >= 0 and initial[r] else np.zeros_like(history[0])
        sequence = np.concatenate([past, x[begin:end]])
        updated[r] = sequence[len(sequence) - state_len :]
        if slot < 0:
            continue
        for p in range(end - begin):
            accum = np.zeros(x.shape[1], np.float32)
            for k in range(taps):
                accum = accum + sequence[p + k * dilation] * weight[:, k]
            pre = _bf16(accum)
            output[begin + p] = _bf16(pre / (1 + np.exp(-pre)))
        if end > begin:
            history[slot] = updated[r]
    return output, history, updated


def _inputs(rng, lengths, taps, dilation, pad_tokens=0):
    tokens = sum(lengths) + pad_tokens
    state_len = (taps - 1) * dilation
    x = _bf16(rng.normal(size=(tokens, CHANNELS)))
    weight = _bf16(rng.normal(size=(CHANNELS, taps)) * 0.5)
    history = _bf16(rng.normal(size=(SLOTS, state_len, CHANNELS)))
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    return x, weight, history, starts


def _run(x, weight, history, starts, slots, initial, dilation, owned=False):
    out, hist = short_conv(
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(weight, jnp.bfloat16),
        jnp.asarray(history, jnp.bfloat16),
        jnp.asarray(starts),
        jnp.asarray(slots, jnp.int32),
        jnp.asarray(initial),
        dilation=dilation,
        request_owned_history=owned,
    )
    return (np.asarray(a.astype(jnp.float32)) for a in (out, hist))


def _close(actual, expected):
    # Only the FP32 accumulation order may differ: one BF16 ulp at most.
    np.testing.assert_allclose(actual, expected, rtol=2**-7, atol=2**-7)


@pytest.mark.parametrize("taps,dilation", [(4, 1), (3, 2), (2, 4)])
def test_short_conv_matches_reference(taps, dilation):
    rng = np.random.default_rng(taps * 10 + dilation)
    # Requests: long, shorter than the state, empty, padded (-1), fresh.
    lengths = [11, 1, 0, 3, 5]
    slots = np.array([4, 0, 2, -1, 1], np.int32)
    initial = np.array([True, True, True, True, False])
    x, weight, history, starts = _inputs(rng, lengths, taps, dilation, pad_tokens=3)
    out, hist = _run(x, weight, history, starts, slots, initial, dilation)
    ref_out, ref_hist, _ = _reference(
        x, weight, history, starts, slots, initial, dilation
    )
    _close(out, ref_out)
    # Tokens of padded requests and past the last start are zero.
    assert not out[starts[3] : starts[4]].any() and not out[starts[-1] :].any()
    _close(hist, ref_hist)
    # Empty, padded and unused slots are untouched.
    for slot in (2, 3, 5):
        np.testing.assert_array_equal(hist[slot], history[slot])


def test_short_conv_request_owned_history():
    rng = np.random.default_rng(7)
    lengths = [6, 2, 9]
    slots = np.array([0, 1, 2], np.int32)  # identity slots
    initial = np.array([True, False, True])
    x, weight, history, starts = _inputs(rng, lengths, 4, 1)
    out, updated = _run(x, weight, history, starts, slots, initial, 1, owned=True)
    ref_out, _, ref_updated = _reference(x, weight, history, starts, slots, initial, 1)
    _close(out, ref_out)
    assert updated.shape == (len(slots), 3, CHANNELS)
    _close(updated, ref_updated)


def test_short_conv_carries_state_across_steps():
    """Two consecutive steps equal one step over the concatenated tokens."""
    rng = np.random.default_rng(3)
    taps, dilation = 3, 2
    slots = np.array([3, 5], np.int32)
    x, weight, history, _ = _inputs(rng, [14, 9], taps, dilation)
    whole = np.array([0, 14, 23], np.int32)
    initial = np.array([True, False])
    ref_out, ref_hist = _run(x, weight, history, whole, slots, initial, dilation)

    first = np.array([0, 8, 12], np.int32)
    x1 = np.concatenate([x[0:8], x[14:18]])
    out1, hist1 = _run(x1, weight, history, first, slots, initial, dilation)
    second = np.array([0, 6, 11], np.int32)
    x2 = np.concatenate([x[8:14], x[18:23]])
    out2, hist2 = _run(
        x2, weight, hist1, second, slots, np.array([True, True]), dilation
    )
    _close(np.concatenate([out1[0:8], out2[0:6]]), ref_out[0:14])
    _close(np.concatenate([out1[8:12], out2[6:11]]), ref_out[14:23])
    _close(hist2, ref_hist)


def test_short_conv_empty_step_is_identity():
    rng = np.random.default_rng(1)
    _, weight, history, _ = _inputs(rng, [1], 4, 1)
    x = np.zeros((0, CHANNELS), np.float32)
    out, hist = _run(x, weight, history, np.zeros(3, np.int32), [0, 1], [True, True], 1)
    assert out.shape == (0, CHANNELS)
    np.testing.assert_array_equal(hist, history)
