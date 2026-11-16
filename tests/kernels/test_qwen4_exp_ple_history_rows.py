# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_HISTORY_ROW_KERNEL: the Pallas history row copy equals XLA's gathers bitwise.

Runs on CPU in TPU interpret mode. Every call is jitted, as in serving: the
HBM pin on the kernel's input does not run eagerly. Each history reader is
compared flag off against flag on: the short conv (output and final window),
the token-major checkpoints and the request-major windows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import history_rows as hr
from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv
from vllm_torchtpu.kernels.qwen4_exp.pool import (
    request_checkpoints,
    speculative_history_windows,
    token_checkpoints,
)

FLAG = "VLLM_TPU_PLE_HISTORY_ROW_KERNEL"
STATE_LEN, WINDOW, CHANNELS, DILATION = 9, 4, 256, 3


def _bits(a):
    return np.asarray(a).view(np.uint16)


def _history(rng, count):
    # Random bit patterns (NaNs included) catch any value-level rewrite.
    bits = rng.integers(0, 1 << 16, size=(count, STATE_LEN, CHANNELS), dtype=np.uint16)
    return jnp.asarray(bits.view(jnp.bfloat16))


def _interpret():
    return pltpu.force_tpu_interpret_mode(pltpu.InterpretParams())


@pytest.mark.parametrize("shape", [(1,), (31,), (32,), (77,), (6, 9), (3, 4, 9)])
def test_rows_match_numpy_indexing(shape):
    rng = np.random.default_rng(sum(shape))
    count = 5
    history = _history(rng, count)
    rows = rng.integers(0, count * STATE_LEN, size=shape).astype(np.int32)
    with _interpret():
        got = jax.jit(hr.gather_history_rows)(history, jnp.asarray(rows))
    want = _bits(history).reshape(count * STATE_LEN, CHANNELS)[rows]
    assert got.shape == shape + (CHANNELS,)
    np.testing.assert_array_equal(_bits(got), want)


def _case(rng, lengths, windowed, ids_on, initial, tokens):
    count = len(lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    assert starts[-1] <= tokens
    ids = np.where(ids_on, 1 + np.arange(count), 0).astype(np.int32)
    ckpt = (1 + np.arange(count * WINDOW, dtype=np.int32)).reshape(count, WINDOW)
    x = jnp.asarray(rng.normal(size=(tokens, CHANNELS)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(count, STATE_LEN, CHANNELS)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(CHANNELS, 4)), jnp.bfloat16)
    starts = jnp.asarray(starts)
    lengths = starts[1:] - starts[:-1]
    active = (jnp.asarray(ids) > 0) & (lengths > 0)
    slots = jnp.where(active, jnp.arange(count, dtype=jnp.int32), -1)
    return dict(
        x=x,
        history=history,
        weight=weight,
        starts=starts,
        initial=jnp.asarray(initial),
        active=active,
        slots=slots,
        ckpt=jnp.asarray(ckpt),
        windowed=jnp.asarray(windowed),
    )


def _readers(c):
    output, final = short_conv(
        c["x"], c["weight"], c["history"], c["starts"], c["slots"], c["initial"],
        dilation=DILATION, request_owned_history=True,
    )
    tok_values, tok_targets = token_checkpoints(
        c["x"], c["history"], c["starts"], c["initial"], c["active"], c["ckpt"],
        c["windowed"],
    )
    req_values, req_targets = request_checkpoints(
        c["x"], c["history"], final, c["starts"], c["initial"], c["active"],
        c["ckpt"], c["windowed"],
    )
    windows = speculative_history_windows(
        c["x"], c["history"], c["starts"], c["initial"], WINDOW
    )
    return dict(
        output=output,
        final=final,
        tok_values=tok_values,
        tok_targets=tok_targets,
        req_values=req_values,
        req_targets=req_targets,
        windows=windows,
    )


def _run(monkeypatch, flag, case):
    monkeypatch.setattr(envs, FLAG, flag)
    with _interpret():
        # A fresh jit per flag value: the flag is read while tracing.
        return jax.block_until_ready(jax.jit(lambda c: _readers(c))(case))


@pytest.mark.parametrize(
    "lengths,windowed,ids_on,initial,tokens",
    [
        # verify windows, one idle row, one short row, padded token tail
        ([4, 4, 0, 2, 3], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 1, 0], 16),
        # decode rows (one token each)
        ([1, 1, 1, 1], [1] * 4, [1, 0, 1, 1], [1, 1, 0, 1], 4),
        # leading windows then a prefill row with and without history
        ([4, 3, 6, 5], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 1], 18),
        # a prefill row shorter than the state, idle prefill, zero-length windowed
        ([2, 0, 3, 7], [0, 1, 0, 0], [1, 1, 0, 1], [0, 1, 1, 1], 12),
        # more gathered rows than one 32-row block, padded request rows
        ([4] * 9 + [0] * 3, [1] * 12, [1] * 9 + [0] * 3, [1] * 12, 40),
    ],
)
def test_flag_on_matches_xla_gathers(monkeypatch, lengths, windowed, ids_on, initial, tokens):
    case = _case(
        np.random.default_rng(len(lengths) * 7 + tokens),
        lengths,
        np.asarray(windowed, bool),
        np.asarray(ids_on, bool),
        np.asarray(initial, bool),
        tokens,
    )
    off = _run(monkeypatch, False, case)
    on = _run(monkeypatch, True, case)
    for name in off:
        a, b = np.asarray(off[name]), np.asarray(on[name])
        if a.dtype == jnp.bfloat16:
            a, b = a.view(np.uint16), b.view(np.uint16)
        np.testing.assert_array_equal(b, a, err_msg=name)
