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
compiled kernel on a TPU). The TPU tests at the served shape compare the
kernel with the served head program compiled the way torch_tpu compiles it
(without excess precision): XLA's windows, random rows, and rows aimed at
every column of the shard with planted exact ties.
"""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.layers.core import draft_propose_split as split
from vllm_torchtpu.layers.core.draft_propose_split import DP_AXIS, TP_AXIS

HIDDEN = 256
WINDOW = 128
# Per TP shard: 97 lane tiles, so the slices of 8 ranks start inside a tile
# and the last ranks' read windows are clamped to the shard's end.
COLS = 12416


def _no_excess_precision(fn, *args):
    """``fn`` compiled for ``args`` as torch_tpu compiles the served program:
    without excess precision, so bf16 results are rounded before use."""
    lowered = jax.jit(fn).lower(*args)
    return lowered.compile(compiler_options={"xla_allow_excess_precision": False})


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

    def argmax(x, w):
        logits = jnp.einsum(
            "dph,thc->dptc", x, w, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        ids = jnp.arange(tp_size * cols).reshape(tp_size, cols)
        logits = jnp.where(ids < vocab, logits.astype(jnp.float32), -jnp.inf)
        first = jnp.argmax(logits, axis=-1)  # [dp, p, tp]
        best = jnp.max(logits, axis=-1)
        shard = jnp.argmax(best, axis=-1)  # [dp, p]
        col = jnp.take_along_axis(first, shard[..., None], axis=-1)[..., 0]
        return shard * cols + col

    return np.asarray(_no_excess_precision(argmax, x, w)(x, w))


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
    operands = (
        xs,
        ws,
        starts.astype(jnp.int32),
        vstart.astype(jnp.int32),
        vend.astype(jnp.int32),
    )
    out = np.asarray(_no_excess_precision(run, *operands)(*operands))
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


def _special_inputs(dp_size, tp_size, num_rows):
    """Integer rows and weights with special logits: engine 0's row 0 has
    +inf in feature 0 (logits +-inf, NaN where that weight is 0), the last
    engine's row 0 is NaN (every logit NaN), and engine 0's last row is +inf
    in feature 1 alone, whose weights are -1 but in two columns 0 (every
    logit -inf but a NaN in column 3 and one past the vocabulary end)."""
    x, w = _integer_inputs(dp_size, tp_size, num_rows, seed=11, ties=False)
    x, w = np.asarray(x, np.float32), np.asarray(w, np.float32)
    x[0, 0, 0] = np.inf
    x[-1, 0] = np.nan
    w[:, 1] = -1
    w[0, 1, 3] = w[-1, 1, COLS - 7] = 0
    x[0, -1] = 0
    x[0, -1, 1] = np.inf
    return jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)


def _first_max_nan_first(x, w, vocab):
    """Per row, the first column of the largest bf16 logit over the whole
    vocabulary, a NaN above every number (torch.max), where ids at or past
    ``vocab`` count as -inf. x [dp, p, H], w [tp, H, cols]; returns
    [dp, p]."""
    tp_size, _, cols = w.shape

    def project(x, w):
        return jnp.einsum(
            "dph,thc->dptc", x, w, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)

    logits = np.asarray(_no_excess_precision(project, x, w)(x, w), np.float32)
    logits = logits.reshape(*logits.shape[:2], tp_size * cols)
    # numpy's argmax takes the first NaN.
    return np.where(np.arange(tp_size * cols) < vocab, logits, -np.inf).argmax(-1)


def _check_special_logits(dp_size, tp_size):
    mesh = _mesh(dp_size, tp_size)
    x, w = _special_inputs(dp_size, tp_size, 6)
    vocab = tp_size * COLS - 300
    got = _split(x, w, vocab, mesh)
    want = _first_max_nan_first(x, w, vocab)
    np.testing.assert_array_equal(got, want)
    assert (got >= 0).all() and (got < vocab).all()
    assert got[-1, 0] == 0 and got[0, -1] == 3


@pytest.mark.cpu_test
@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (8, 1)])
def test_nan_logits_win_first_cpu(small_windows, dp_size, tp_size):
    """+-inf and NaN logits: a NaN counts as larger than every number, the
    first one winning, as in torch.max, and every token is a vocabulary
    id."""
    _check_special_logits(dp_size, tp_size)


@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (8, 1)])
def test_nan_logits_win_first(small_windows, dp_size, tp_size):
    if jax.default_backend() != "tpu":
        pytest.skip("compiled kernel on a TPU")
    _check_special_logits(dp_size, tp_size)


@pytest.mark.cpu_test
@pytest.mark.parametrize("cols", [62080, COLS])
def test_read_window_covers_each_slice_inside_the_shard(cols):
    for dp_size in (2, 4, 8):
        read = split.read_columns(cols, dp_size)
        assert read % split._BLOCK_COLS == 0 and read <= cols
        for d in range(dp_size):
            start = d * (cols // dp_size)
            first = min(start // 128 * 128, cols - read)
            assert first % 128 == 0
            assert first <= start and start + cols // dp_size <= first + read


@pytest.mark.cpu_test
def test_supported_row_counts():
    """At the served shape the window table keeps 1-56 draft rows per engine
    at DP 2-8 (the VMEM budget would take 62 at DP8); other shapes and a
    single engine keep the vocab-parallel argmax."""
    for dp_size in (8, 4, 2):
        rows = [r for r in range(1, 257) if split.supported(r, 8192, 62080, dp_size)]
        assert rows == list(range(1, 57)), dp_size
    assert split._vmem_bytes(8 * 62, 8192, 512) <= split._VMEM_BUDGET
    assert split._vmem_bytes(8 * 63, 8192, 512) > split._VMEM_BUDGET
    assert split.xla_head_window(26, 8192, 62080) == 1024
    assert split.xla_head_window(56, 8192, 62080) == 512
    assert split.xla_head_window(57, 8192, 62080) is None
    assert not split.supported(26, 8192, 62080, 1)
    assert not split.supported(26, 8192, 62081, 8)
    assert not split.supported(26, 4096, 62080, 8)


# Served shape: one TP4 shard of the 248320-token head, 8 engines.
_H, _COLS, _DP = 8192, 62080, 8
_SLICE = _COLS // _DP
_VOCAB_START = 3 * _COLS


def _skip_off_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("XLA's TPU dot")


def _served_pair(x, w, vocab_start, vocab_end, *, form):
    """The draft head program on one shard as torch_tpu builds it: the
    hidden-major projection and local_argmax_pair (the masked maximum and its
    first column as a global id), with the shard's bounds as operands. The
    maximum's reduce body is torch_tpu's for torch.max: "ge" (builds before
    2026-09-30) keeps the left value where it is >= the right one, "max"
    (later builds) takes jax.lax.max and lets a NaN win; equal values (and
    two NaNs) take the smaller int64 index."""
    logits = jnp.dot(x, w)
    ids = jax.lax.iota(jnp.int32, w.shape[1]) + vocab_start[0]
    logits = jnp.where(ids < vocab_end[0], logits, jnp.array(-jnp.inf, logits.dtype))
    iota = jax.lax.broadcasted_iota(jnp.int64, logits.shape, 1)

    def body(a, b):
        ge, eq = a[0] >= b[0], a[0] == b[0]
        if form == "max":
            a_nan, b_nan = a[0] != a[0], b[0] != b[0]
            ge, eq = ge | (a_nan & ~b_nan), eq | (a_nan & b_nan)
            value = jax.lax.max(a[0], b[0])
        else:
            value = jnp.where(ge, a[0], b[0])
        index = jnp.where(eq, jnp.minimum(a[1], b[1]), jnp.where(ge, a[1], b[1]))
        return value, index

    init = (jnp.array(jnp.finfo(logits.dtype).min, logits.dtype), jnp.int64(0))
    value, col = jax.lax.reduce((logits, iota), init, body, (1,))
    col = col.astype(jnp.int32) + vocab_start[0]
    return jnp.stack([value.astype(jnp.float32), col.astype(jnp.float32)], -1)


def _served_shapes(num_rows):
    return (
        jax.ShapeDtypeStruct((num_rows, _H), jnp.bfloat16),
        jax.ShapeDtypeStruct((_H, _COLS), jnp.bfloat16),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )


@functools.cache
def _served_head(num_rows, form="max"):
    with jax.enable_x64(True):
        return _no_excess_precision(
            functools.partial(_served_pair, form=form), *_served_shapes(num_rows)
        )


@functools.cache
def _slice_kernel(num_rows):
    window = split.xla_head_window(num_rows, _H, _COLS)
    run = functools.partial(
        split.slice_argmax, read_cols=split.read_columns(_COLS, _DP)
    )
    return _no_excess_precision(
        run,
        jax.ShapeDtypeStruct((_H // window, _DP * num_rows, window), jnp.bfloat16),
        jax.ShapeDtypeStruct((_H, _COLS), jnp.bfloat16),
        jax.ShapeDtypeStruct((4,), jnp.int32),
    )


def _served_pairs(x, w, vocab_end):
    """The served program on each engine's rows of x [DP * p, H]: values
    [DP * p] float32 and global ids."""
    num_rows = x.shape[0] // _DP
    run = _served_head(num_rows)
    bounds = (
        jnp.asarray([_VOCAB_START], jnp.int32),
        jnp.asarray([vocab_end], jnp.int32),
    )
    pairs = np.concatenate(
        [
            np.asarray(run(x[e * num_rows : (e + 1) * num_rows], w, *bounds))
            for e in range(_DP)
        ]
    )
    return pairs[:, 0], pairs[:, 1].astype(np.int64)


def _kernel_pairs(x, w, vocab_end):
    """Every engine's rows x [DP * p, H] through the slice kernel of each DP
    rank, combined as the exchange combines them: per row the first maximum
    in vocabulary order. Returns values [DP * p] float32 and global ids."""
    num_rows = x.shape[0] // _DP
    window = split.xla_head_window(num_rows, _H, _COLS)
    read = split.read_columns(_COLS, _DP)
    rows = x.reshape(_DP * num_rows, _H // window, window).transpose(1, 0, 2)
    run = _slice_kernel(num_rows)
    values, ids = [], []
    for d in range(_DP):
        start = d * _SLICE
        first = min(start // 128 * 128, _COLS - read)
        meta = jnp.asarray(
            [
                first,
                start - first,
                start - first + _SLICE,
                vocab_end - _VOCAB_START - first,
            ],
            jnp.int32,
        )
        pair = np.asarray(run(rows, w, meta))
        values.append(pair[:, 0])
        ids.append(pair[:, 1].astype(np.int64) + first + _VOCAB_START)
    values, ids = np.stack(values, 1), np.stack(ids, 1)
    pick = values.argmax(1)[:, None]  # the first maximum: the lowest slice
    return (
        np.take_along_axis(values, pick, 1)[:, 0],
        np.take_along_axis(ids, pick, 1)[:, 0],
    )


def _twin_pairs(seed):
    """Disjoint column pairs (a, b), a < b, that get equal weights: (e - 1, e)
    across every slice edge, the edges of XLA's column windows and of their
    split tiles, the edges of the read windows' blocks and the shard's last
    lane tile, then random pairs, until one column in four is paired. Returns
    the pairs [n, 2] and how many of them, first, are edge pairs."""
    read = split.read_columns(_COLS, _DP)
    firsts = [min(d * _SLICE // 128 * 128, _COLS - read) for d in range(_DP)]
    blocks = read // split._BLOCK_COLS
    edges = sorted(
        {d * _SLICE for d in range(1, _DP)}
        | {
            k * n + s
            for n in (6272, 6912, 12416)
            for k in range(11)
            for s in (-128, 0, 256)
        }
        | {f + b * split._BLOCK_COLS for f in firsts for b in range(blocks + 1)}
        | {_COLS - 128, _COLS - 1}
    )
    used, pairs = set(), []
    for e in edges:
        if 0 < e < _COLS and not {e - 1, e} & used:
            pairs.append((e - 1, e))
            used |= {e - 1, e}
    num_edge = len(pairs)
    free = np.setdiff1d(np.arange(_COLS), np.asarray(sorted(used)))
    free = np.random.default_rng(seed).permutation(free)
    for a, b in free[: 2 * (_COLS // 8 - num_edge)].reshape(-1, 2):
        pairs.append((min(a, b), max(a, b)))
    return np.asarray(pairs, np.int64), num_edge


def _first_copies(pairs):
    """Per column, the first column holding the same weights."""
    first = np.arange(_COLS)
    first[pairs[:, 1]] = pairs[:, 0]
    return first


def _weights(seed, pairs):
    """[H, cols] bf16 N(0, 0.02) weights; column b copies column a for each
    (a, b) in ``pairs``."""
    w = jax.random.normal(jax.random.key(seed), (_H, _COLS), jnp.float32)
    w = (w * 0.02).astype(jnp.bfloat16)
    if len(pairs):
        w = w.at[:, pairs[:, 1]].set(w[:, pairs[:, 0]])
    return w


def _aimed_rows(key, w, columns):
    """Rows aimed at ``columns``: 8 x the column's weights plus N(0, 1)
    noise. The aimed column's logit (about 26) leads every other column's
    (below about 10, its copy aside), and its sum cancels like a random
    row's."""
    noise = jax.random.normal(key, (columns.size, _H), jnp.float32)
    return (8 * w[:, columns].T.astype(jnp.float32) + noise).astype(jnp.bfloat16)


def _split_tile_columns(num_rows):
    """Columns whose sum XLA's head dot splits over both MXUs at this row
    count on libtpu 0.0.47: the first 256 of each 6912-column window up to 40
    rows and the last 128 of each 12416-column window above. Every other
    column adds one MXU sum per window to the running sum, in window order,
    as the kernel does."""
    cols = np.arange(_COLS)
    if num_rows <= 40:
        return cols % 6912 < 256
    return cols % 12416 >= 12288


def _assert_within_one_bf16_unit(got, want):
    """Finite values that differ by at most one bf16 unit in the last place
    of the larger magnitude: what summing in another order can change."""
    assert np.isfinite(got).all() and np.isfinite(want).all()
    mag = np.maximum(np.maximum(np.abs(got), np.abs(want)), np.finfo(np.float32).tiny)
    far = np.abs(got - want) > np.exp2(np.floor(np.log2(mag)) - 7)
    assert not far.any(), (np.flatnonzero(far)[:8], got[far][:8], want[far][:8])


@pytest.mark.parametrize("form", ["ge", "max"])
@pytest.mark.parametrize("num_rows", [1, 2, 26, 40, 41, 52, 56])
def test_window_table_matches_the_served_head(num_rows, form):
    """XLA's dot in the served program, with either torch_tpu reduce body,
    sums the hidden dimension in the table's windows on this libtpu."""
    _skip_off_tpu()
    text = _served_head(num_rows, form).as_text()
    found = [
        (re.search(r'"kernel_window_bounds":\["(\d+)"', line), line)
        for line in text.splitlines()
        if "kind=kOutput" in line
    ]
    found = [(m, line) for m, line in found if m]
    assert len(found) == 1, "re-derive _XLA_HEAD_WINDOWS: new head fusion layout"
    ((bounds, line),) = found
    iters = re.search(r'"iteration_bounds":\[([^\]]*)\]', line).group(1)
    by_bounds = int(bounds.group(1)) * 8
    by_iters = _H // int(iters.split(",")[-1].strip('"'))
    window = split.xla_head_window(num_rows, _H, _COLS)
    assert by_bounds == by_iters == window, (by_bounds, by_iters, window)


@pytest.mark.parametrize("num_rows", [1, 8, 23, 24, 26, 40, 41, 46, 52, 56])
def test_slice_kernel_matches_the_served_head(num_rows):
    """Random rows (near ties at the top) and rows aimed at the planted
    copies across the edges: the eight slices' pairs, combined, are the
    served program's up to the order of its sums, and every aimed row's token
    is the first copy."""
    _skip_off_tpu()
    assert split.supported(num_rows, _H, _COLS, _DP)
    pairs, num_edge = _twin_pairs(seed=num_rows)
    first = _first_copies(pairs)
    w = _weights(num_rows, pairs)
    vocab_end = _VOCAB_START + _COLS - 1000
    key = jax.random.key(1000 + num_rows)
    batch = _DP * num_rows
    for b in range(4):
        x = jax.random.normal(jax.random.fold_in(key, b), (batch, _H), jnp.float32)
        x = x.astype(jnp.bfloat16)
        kv, kid = _kernel_pairs(x, w, vocab_end)
        sv, sid = _served_pairs(x, w, vocab_end)
        _assert_within_one_bf16_unit(kv, sv)
        assert (kid < vocab_end).all() and (sid < vocab_end).all()
    edge = pairs[:num_edge, 0]
    edge = edge[pairs[:num_edge, 1] < vocab_end - _VOCAB_START]
    for b0 in range(0, edge.size, batch):
        columns = edge[np.arange(b0, b0 + batch) % edge.size]
        x = _aimed_rows(jax.random.fold_in(key, 100 + b0), w, columns)
        kv, kid = _kernel_pairs(x, w, vocab_end)
        sv, sid = _served_pairs(x, w, vocab_end)
        np.testing.assert_array_equal(kid, columns + _VOCAB_START)
        np.testing.assert_array_equal(first[sid - _VOCAB_START], columns)
        _assert_within_one_bf16_unit(kv, sv)


@pytest.mark.parametrize("num_rows", [26, 52])
def test_every_column_decides_a_row_with_the_served_head_bits(num_rows):
    """Rows aimed at every column of the shard, four noise draws each, with
    one column in four an exact copy of another: the kernel's token is always
    the first copy and the served program's a copy, the values agree within
    one bf16 unit, and wherever XLA builds the sums as the kernel does
    (outside its split tiles) values and tokens are bitwise the served
    program's."""
    _skip_off_tpu()
    pairs, _ = _twin_pairs(seed=7 * num_rows)
    first = _first_copies(pairs)
    tiles = _split_tile_columns(num_rows)
    in_tiles = tiles.copy()
    in_tiles[pairs[:, 0]] |= tiles[pairs[:, 1]]
    in_tiles[pairs[:, 1]] |= tiles[pairs[:, 0]]
    w = _weights(7 * num_rows, pairs)
    vocab_end = _VOCAB_START + _COLS
    key = jax.random.key(num_rows)
    batch = _DP * num_rows
    tile_rows = tile_diffs = 0
    for draw in range(4):
        for b0 in range(0, _COLS, batch):
            columns = np.arange(b0, b0 + batch) % _COLS
            x = _aimed_rows(jax.random.fold_in(key, draw * _COLS + b0), w, columns)
            kv, kid = _kernel_pairs(x, w, vocab_end)
            sv, sid = _served_pairs(x, w, vocab_end)
            np.testing.assert_array_equal(kid, first[columns] + _VOCAB_START)
            np.testing.assert_array_equal(first[sid - _VOCAB_START], first[columns])
            _assert_within_one_bf16_unit(kv, sv)
            alike = ~in_tiles[columns]
            np.testing.assert_array_equal(kv[alike], sv[alike])
            np.testing.assert_array_equal(sid[alike], kid[alike])
            tile_rows += int((~alike).sum())
            tile_diffs += int((kv[~alike] != sv[~alike]).sum())
    print(
        f"rows {num_rows}: split-tile rows {tile_rows}, value differences {tile_diffs}"
    )
