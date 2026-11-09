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
"""Greedy draft tokens from the LM head split over DP against the
vocab-parallel argmax.

The mesh tests use small integer values, so every logit is an exact sum and
any summation order gives the same bf16 bits: they check the slices, the
exchange and the tie-breaking (on the CPU in interpret mode, and with the
compiled kernel on a TPU). The TPU tests at the served shape check that the
kernel's windows reproduce XLA's dot bit for bit.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.layers.core import draft_propose_split as split
from vllm_torchtpu.layers.core.draft_propose_split import DP_AXIS, TP_AXIS

HIDDEN = 256
WINDOW = 128
COLS = 4096  # per TP shard


def _mesh(dp_size, tp_size):
    devices = jax.devices()
    if len(devices) < dp_size * tp_size:
        pytest.skip(f"needs {dp_size * tp_size} devices")
    grid = np.asarray(devices[: dp_size * tp_size]).reshape(dp_size, tp_size)
    return Mesh(grid, (DP_AXIS, TP_AXIS))


@pytest.fixture
def small_windows(monkeypatch):
    table = dict(split._XLA_HEAD_WINDOWS)
    for rows in range(1, 33):
        table[(rows, HIDDEN, COLS)] = WINDOW
    monkeypatch.setattr(split, "_XLA_HEAD_WINDOWS", table)


def _interpret():
    return jax.default_backend() != "tpu"


def _reference(x, w, vocab):
    """The vocab-parallel argmax: per TP shard the first maximum of its bf16
    logits (ids at or past ``vocab`` count as -inf), then the first shard
    holding the largest. x [dp, p, H], w [tp, H, cols]; returns [dp, p]."""
    tp_size, _, cols = w.shape
    logits = jnp.einsum(
        "dph,thc->dptc", x, w, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    ids = jnp.arange(tp_size * cols).reshape(tp_size, cols)
    logits = jnp.where(ids < vocab, logits.astype(jnp.float32), -jnp.inf)
    first = jnp.argmax(logits, axis=-1)  # [dp, p, tp]
    best = jnp.max(logits, axis=-1)
    shard = jnp.argmax(best, axis=-1)  # [dp, p]
    col = jnp.take_along_axis(first, shard[..., None], axis=-1)[..., 0]
    return np.asarray(shard * cols + col)


def _split(x, w, vocab, mesh):
    dp_size, tp_size = mesh.shape[DP_AXIS], mesh.shape[TP_AXIS]
    num_rows = x.shape[1]
    cols = w.shape[-1]
    xs = jnp.broadcast_to(x[:, None], (dp_size, tp_size, num_rows, HIDDEN))
    ws = jnp.broadcast_to(w[None], (dp_size, tp_size, HIDDEN, cols))
    d = np.arange(dp_size)[:, None, None]
    t = np.arange(tp_size)[None, :, None]
    starts = jnp.asarray(np.broadcast_to(d * (cols // dp_size), (dp_size, tp_size, 1)))
    vstart = jnp.asarray(np.broadcast_to(t * cols, (dp_size, tp_size, 1)))
    vend = jnp.asarray(
        np.broadcast_to(np.minimum((t + 1) * cols, vocab), (dp_size, tp_size, 1))
    )
    run = functools.partial(split.draft_argmax, mesh=mesh, interpret=_interpret())
    out = jax.jit(run)(
        xs,
        ws,
        starts.astype(jnp.int32),
        vstart.astype(jnp.int32),
        vend.astype(jnp.int32),
    )
    out = np.asarray(out)
    # Every TP rank of an engine returns the engine's tokens.
    for t in range(1, tp_size):
        np.testing.assert_array_equal(out[:, t], out[:, 0])
    return out[:, 0]


def _integer_inputs(dp_size, tp_size, num_rows, seed, ties=True):
    rng = np.random.default_rng(seed)
    x = rng.integers(-2, 3, size=(dp_size, num_rows, HIDDEN)).astype(np.float32)
    w = rng.integers(-2, 3, size=(tp_size, HIDDEN, COLS)).astype(np.float32)
    if ties:
        slice_cols = COLS // dp_size
        # Row 0 of engine 0: the same column at the end of slice 0, the start
        # of slice 1 and in the last shard -> the first one must win.
        col = rng.integers(-2, 3, size=HIDDEN).astype(np.float32)
        x[0, 0] = col * 3
        tied = [(0, slice_cols - 1), (0, slice_cols)]
        if tp_size > 1:
            tied.append((tp_size - 1, 5))
        for t, c in tied:
            w[t, :, c] = col
        # Row 1 of the last engine: a tie inside one slice of the last
        # shard, at its last two columns.
        col = rng.integers(-2, 3, size=HIDDEN).astype(np.float32)
        x[-1, min(1, num_rows - 1)] = col * 3
        for c in (COLS - 2, COLS - 1):
            w[tp_size - 1, :, c] = col
    if num_rows > 2:
        x[:, -1] = 0  # padding rows: every logit ties at zero
    return jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)


@pytest.mark.cpu_test
@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize("num_rows", [1, 5, 26])
def test_split_tokens_are_the_vocab_parallel_argmax_cpu(
    small_windows, dp_size, tp_size, num_rows
):
    mesh = _mesh(dp_size, tp_size)
    x, w = _integer_inputs(dp_size, tp_size, num_rows, seed=num_rows + dp_size)
    vocab = tp_size * COLS
    np.testing.assert_array_equal(_split(x, w, vocab, mesh), _reference(x, w, vocab))
    if num_rows > 1:
        assert _split(x, w, vocab, mesh)[0, 0] == COLS // dp_size - 1


@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize("num_rows", [5, 26])
def test_split_tokens_are_the_vocab_parallel_argmax(
    small_windows, dp_size, tp_size, num_rows
):
    if jax.default_backend() != "tpu":
        pytest.skip("compiled kernel on a TPU")
    mesh = _mesh(dp_size, tp_size)
    x, w = _integer_inputs(dp_size, tp_size, num_rows, seed=num_rows * dp_size)
    vocab = tp_size * COLS
    np.testing.assert_array_equal(_split(x, w, vocab, mesh), _reference(x, w, vocab))


@pytest.mark.cpu_test
@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (8, 1)])
def test_ids_at_or_past_the_vocabulary_end_never_win(small_windows, dp_size, tp_size):
    """The last shard's tail is padding: its columns hold the largest
    logits, yet the tokens stay below the vocabulary size."""
    mesh = _mesh(dp_size, tp_size)
    x, w = _integer_inputs(dp_size, tp_size, 6, seed=3, ties=False)
    vocab = tp_size * COLS - 300
    w = w.at[-1, :, COLS - 300 :].set(2)
    x = x.at[:, :, :].set(jnp.abs(x))
    want = _reference(x, w, vocab)
    got = _split(x, w, vocab, mesh)
    np.testing.assert_array_equal(got, want)
    assert (got < vocab).all()


@pytest.mark.cpu_test
def test_read_window_covers_each_slice_inside_the_shard():
    cols = 62080
    for dp_size in (2, 4, 8):
        read = split.read_columns(cols, dp_size)
        assert read % 2048 == 0 and read <= cols
        for d in range(dp_size):
            start = d * (cols // dp_size)
            first = min(start // 128 * 128, cols - read)
            assert first % 128 == 0
            assert first <= start and start + cols // dp_size <= first + read


@pytest.mark.cpu_test
def test_supported_row_counts():
    assert split.supported(26, 8192, 62080, 8)
    assert split.xla_head_window(26, 8192, 62080) == 1024
    assert split.xla_head_window(52, 8192, 62080) == 512
    assert not split.supported(26, 8192, 62080, 1)
    assert not split.supported(26, 8192, 62081, 8)
    assert not split.supported(26, 4096, 62080, 8)
    assert not split.supported(400, 8192, 62080, 8)


# Served shape: one TP4 shard of the 248320-token head, 8 engines.
_H, _COLS, _DP = 8192, 62080, 8


def _xla_head_pairs(x, w, vocab_start, vocab_end):
    """The vocab-parallel argmax program on one shard: the hidden-major dot
    fused with the masked maximum and first index, compiled by XLA."""

    def pair(x, w):
        logits = jnp.dot(x, w)
        ids = jax.lax.iota(jnp.int32, w.shape[1]) + vocab_start
        logits = jnp.where(ids < vocab_end, logits, jnp.array(-jnp.inf, logits.dtype))
        iota = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)

        def first_max(a, b):
            take_a = (a[0] > b[0]) | ((a[0] == b[0]) & (a[1] < b[1]))
            return jnp.where(take_a, a[0], b[0]), jnp.where(take_a, a[1], b[1])

        init = (jnp.array(-jnp.inf, logits.dtype), jnp.int32(0))
        value, col = jax.lax.reduce((logits, iota), init, first_max, (1,))
        return value.astype(jnp.float32), col + vocab_start

    return jax.jit(pair)(x, w)


def _served_inputs(num_rows, seed):
    key = jax.random.key(seed)
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (_DP, num_rows, _H), jnp.float32).astype(jnp.bfloat16)
    w = (jax.random.normal(kw, (_H, _COLS), jnp.float32) * 0.02).astype(jnp.bfloat16)
    # Exact ties across the slice boundaries of DP ranks 0 | 1 and 6 | 7.
    for d, r in ((0, 0), (6, 1)):
        c = (d + 1) * (_COLS // _DP)
        w = w.at[:, c].set(w[:, c - 1])
        x = x.at[d, min(r, num_rows - 1)].set(
            (w[:, c].astype(jnp.float32) * 50).astype(jnp.bfloat16)
        )
    return x, w


@pytest.mark.parametrize("num_rows", [8, 23, 24, 26, 40, 41, 46, 52, 80])
def test_slice_kernel_keeps_the_bits_of_xla_head(num_rows):
    """Each engine's draft rows: the eight slices' pairs, combined, equal the
    vocab-parallel argmax program's pair on the whole shard."""
    if jax.default_backend() != "tpu":
        pytest.skip("XLA's TPU dot")
    assert split.supported(num_rows, _H, _COLS, _DP)
    x, w = _served_inputs(num_rows, seed=num_rows)
    vocab_start, vocab_end = 3 * _COLS, 4 * _COLS - 1000
    want_v, want_id = [], []
    for e in range(_DP):
        v, i = _xla_head_pairs(x[e], w, vocab_start, vocab_end)
        want_v.append(np.asarray(v))
        want_id.append(np.asarray(i))
    window = split.xla_head_window(num_rows, _H, _COLS)
    read = split.read_columns(_COLS, _DP)
    rows = x.reshape(_DP * num_rows, _H // window, window).transpose(1, 0, 2)
    run = jax.jit(functools.partial(split.slice_argmax, read_cols=read))
    got = []
    for d in range(_DP):
        start = d * (_COLS // _DP)
        first = min(start // 128 * 128, _COLS - read)
        meta = jnp.asarray(
            [
                first,
                start - first,
                start - first + _COLS // _DP,
                vocab_end - vocab_start - first,
            ],
            jnp.int32,
        )
        pair = np.asarray(run(rows, w, meta))
        got.append((pair[:, 0], pair[:, 1].astype(np.int64) + first + vocab_start))
    values = np.stack([g[0] for g in got], axis=1).reshape(_DP, num_rows, _DP)
    ids = np.stack([g[1] for g in got], axis=1).reshape(_DP, num_rows, _DP)
    pick = values.argmax(axis=-1)
    got_v = np.take_along_axis(values, pick[..., None], axis=-1)[..., 0]
    got_id = np.take_along_axis(ids, pick[..., None], axis=-1)[..., 0]
    for e in range(_DP):
        np.testing.assert_array_equal(got_v[e], want_v[e])
        np.testing.assert_array_equal(got_id[e], want_id[e])
    top2 = np.sort(np.asarray(jnp.dot(x.reshape(-1, _H), w), np.float32), axis=-1)
    ties = int((top2[:, -1] == top2[:, -2]).sum())
    print(f"rows {num_rows}: {ties} of {_DP * num_rows} rows with a tied maximum")
