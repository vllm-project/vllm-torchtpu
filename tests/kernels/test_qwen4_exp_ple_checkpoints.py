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
        x, weight, history, starts, slots, initial, dilation=3,
        request_owned_history=True,
    )
    args = (x, history, starts, initial, active, jnp.asarray(ckpt), jnp.asarray(windowed))
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
def test_token_checkpoints_match_request_checkpoints(lengths, windowed, ids_on, initial, tokens):
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
        np.testing.assert_array_equal(new[target], old[target], err_msg=f"target {target}")


def test_token_checkpoints_catch_an_off_by_one_window():
    """Positive control: shifting the token windows by one must be detected."""
    rng = np.random.default_rng(3)
    old, new = _case(
        rng, [4, 4], np.array([1, 1], bool), np.array([1, 1], bool), np.array([1, 1], bool), 8
    )
    shifted = {k: new[k + 1] for k in new if k + 1 in new}
    assert any(not np.array_equal(shifted[k], old[k]) for k in shifted)
