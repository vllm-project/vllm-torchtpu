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
"""The ragged all-to-all row copy kernels against the XLA ops they replace."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.row_copy import gather_rows
from vllm_torchtpu.layers.core import ragged_a2a_moe
from vllm_torchtpu.layers.core.fused_moe_gmm import onehot_permute
from vllm_torchtpu.layers.core.ragged_a2a_moe import _scatter_send, _tile_rows

# bf16 bit patterns the one-hot permute changes or must keep: +-0,
# subnormals, +-Inf, NaNs, the smallest and largest normals.
_SPECIALS = np.array(
    [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x7F80, 0xFF80, 0x7FC0,
     0xFFC1, 0x7F7F, 0xFF7F, 0x0080, 0x8080],
    np.uint16,
)  # fmt: skip


def _require_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()}.")


def _rows(rng, num_rows, hidden):
    """bf16 rows [num_rows, hidden] with special values in about 1% of the
    elements."""
    x = np.array(
        jnp.asarray(rng.standard_normal((num_rows, hidden)), jnp.bfloat16)
    ).view(np.uint16)
    hit = rng.random(x.shape) < 0.01
    x[hit] = rng.choice(_SPECIALS, size=int(hit.sum()))
    return jnp.asarray(x.view(jnp.bfloat16))


def _group_sizes(rng, count, groups=16):
    """``groups`` sizes that sum to ``count``, like a route plan's."""
    cuts = np.sort(rng.integers(0, count + 1, size=groups - 1))
    return np.diff(np.concatenate([[0], cuts, [count]])).astype(np.int32)


@pytest.mark.parametrize("num_routes", [512, 640, 1024, 1840])
@pytest.mark.parametrize("count", [0, 1, 63, 64, 65, 260, -1])
@pytest.mark.parametrize("hidden", [8192, 2048])
def test_gather_rows_matches_the_onehot_permute(num_routes, count, hidden):
    """Every row the expert kernel reads (the computed routes, padded to whole
    16-row blocks with zeros) is bitwise the one-hot permute's row."""
    _require_tpu()
    count = num_routes if count < 0 else count
    rng = np.random.default_rng(num_routes * 10 + count)
    # Receive rows as served: 184 at the 2-member proxy (1840 routes), 480 in
    # the 4-host capacity exchange.
    num_src = 184 if num_routes == 1840 else 480
    rows2d = _rows(rng, num_src, hidden)
    index = rng.integers(0, num_src, size=num_routes).astype(np.int32)
    # Routes past the computed ones read no row (the plan's out-of-range index).
    index[count:] = num_src
    sizes = _group_sizes(rng, count)
    # The rows the routes read lie below the filled bound, as in the exchange.
    filled = int(index[:count].max()) + 1 if count else 0
    got = gather_rows(
        rows2d.reshape(num_src, 16, hidden // 16),
        jnp.asarray(index),
        jnp.asarray(sizes),
        jnp.asarray([filled], jnp.int32),
    )
    want = jax.jit(onehot_permute)(rows2d, jnp.asarray(index))
    written = min(num_routes, -(-count // 16) * 16)
    got = np.asarray(got).view(np.uint16)[:written]
    want = np.asarray(want).view(np.uint16)[:written]
    np.testing.assert_array_equal(got, want)


def _send_layout(routed, num_members, rows):
    """The capacity exchange's send layout from each route's member [S, K]
    (``num_members``: none): send rows [S, D] (``rows``: not sent) in
    destination order, each destination in token order, the token of each
    send row (the XLA path's index) and the rows the exchange sends."""
    need = (routed[:, :, None] == np.arange(num_members)).any(axis=1)
    sizes = need.sum(axis=0)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out_row = np.where(need, start + np.cumsum(need, axis=0) - 1, rows)
    index = np.zeros(rows, np.int32)
    for t, d in zip(*np.nonzero(need & (out_row < rows))):
        index[out_row[t, d]] = t
    return out_row.astype(np.int32), index, min(int(sizes.sum()), rows)


@pytest.mark.parametrize("num_members", [2, 8, 16, 32])
@pytest.mark.parametrize("num_tokens", [12, 92, 104, 112, 128])
@pytest.mark.parametrize("routing", ["random", "one_member", "none", "overflow"])
def test_copy_send_rows_writes_the_send_rows(num_members, num_tokens, routing):
    """The send buffer rows the exchange sends are exact copies of their
    token rows, in the XLA path's layout."""
    _require_tpu()
    topk = 10
    rng = np.random.default_rng(num_members * 1000 + num_tokens)
    routed = rng.integers(0, num_members + 1, size=(num_tokens, topk))
    if routing == "one_member":
        routed[:] = num_members
        routed[:, 3] = num_members // 2
        routed[::2, 7] = num_members // 2
    elif routing == "none":
        routed[:] = num_members
    need = (routed[:, :, None] == np.arange(num_members)).any(axis=1)
    rows = max(16, -(-int(need.sum()) // 32) * 32)
    if routing == "overflow":
        # Buffers smaller than the step: rows past the buffer are dropped.
        rows = max(16, rows // 2 // 16 * 16)
    out_row, index, total = _send_layout(routed, num_members, rows)
    x = _rows(rng, num_tokens, 8192)
    got = _scatter_send(x, jnp.asarray(out_row), rows=rows)
    # The layout of the XLA path's send buffer, with the token rows copied
    # bit for bit.
    want_shape = jax.eval_shape(lambda x, i: _tile_rows(x[i]), x, jnp.asarray(index))
    assert got.shape == want_shape.shape
    want = np.asarray(x).view(np.uint16)[index].reshape(want_shape.shape)
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16)[:total], want[:total])


def _need(rng, num_tokens, num_members, routing):
    """Whether token t goes to member d [S, D]: ``random`` (each member
    about half the tokens), ``skewed`` (most tokens to member 0, a few to
    the others), ``dense`` (every token to most members), ``unrouted`` (the
    last fifth of the tokens to none) or ``none``."""
    if routing == "none":
        return np.zeros((num_tokens, num_members), bool)
    p = {"random": 0.5, "skewed": 0.05, "dense": 0.9, "unrouted": 0.5}[routing]
    need = rng.random((num_tokens, num_members)) < p
    if routing == "skewed":
        need[:, 0] = rng.random(num_tokens) < 0.95
    if routing == "unrouted":
        need[num_tokens - max(1, num_tokens // 5) :] = False
    return need


def _send_rows(need, rows, layout):
    """The send row of each (token, member) pair [S, D] (``rows``: not
    sent): the capacity exchange's gapless destination groups (positions
    past the buffer are dropped), or the full-size exchange's region of S
    rows per member."""
    num_tokens, num_members = need.shape
    rank = np.cumsum(need, axis=0) - 1
    if layout == "regions":
        start = np.arange(num_members) * num_tokens
    else:
        sizes = need.sum(axis=0)
        start = np.cumsum(sizes) - sizes
    return np.where(need, start + rank, rows).astype(np.int32)


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_members", [2, 8, 16, 32])
@pytest.mark.parametrize("layout", ["capacity", "regions"])
def test_send_index_rows_minor_matches_the_default(monkeypatch, num_members, layout):
    """The index and count `_scatter_send` hands `copy_send_rows` with
    ``rows_minor`` equal the default's on every row: capacity buffers from one
    row to the whole step (smaller than the step: positions past the buffer
    dropped) and the full-size exchange's regions, with empty, partial and
    full destination groups."""
    monkeypatch.setattr(
        ragged_a2a_moe, "copy_send_rows", lambda x, index, count, rows: (index, count)
    )
    send = jax.jit(
        lambda x, out_row, rows, rows_minor: _scatter_send(
            x, out_row, rows=rows, rows_minor=rows_minor
        ),
        static_argnames=("rows", "rows_minor"),
    )
    rng = np.random.default_rng(num_members)
    for num_tokens in (4, 13, 104, 128):
        x = jnp.zeros((num_tokens, 128), jnp.bfloat16)
        slots = num_members * num_tokens
        if layout == "regions":
            row_counts = [slots]
        else:
            row_counts = sorted({1, 2, 15, 16, 17, slots // 3, slots // 2, slots})
        for rows in row_counts:
            for routing in ("random", "skewed", "dense", "unrouted", "none"):
                out_row = jnp.asarray(
                    _send_rows(
                        _need(rng, num_tokens, num_members, routing), rows, layout
                    )
                )
                want = send(x, out_row, rows=rows, rows_minor=False)
                got = send(x, out_row, rows=rows, rows_minor=True)
                np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
                np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))


@pytest.mark.parametrize(
    "num_members,num_tokens,rows",
    [
        # Capacity buffers and full-size regions (rows = members x tokens);
        # at 128 tokens the default sum compiles with the send rows major.
        (8, 104, 480),
        (8, 104, 832),
        (8, 128, 912),
        (16, 112, 544),
        (16, 112, 1056),
        (16, 128, 1216),
        (16, 128, 2048),
    ],
)
def test_send_index_rows_minor_matches_the_default_on_tpu(
    monkeypatch, num_members, num_tokens, rows
):
    """Compiled for TPU at the exchange's 8- and 16-member sizes, the send
    index and count with ``rows_minor`` are the default's on every row, for
    uniform, skewed, overflowing, partly unrouted and empty steps, and the
    send rows written are the same bits."""
    _require_tpu()
    rng = np.random.default_rng(rows)
    layout = "regions" if rows == num_members * num_tokens else "capacity"
    x = _rows(rng, num_tokens, 8192)

    def compiled(rows_minor):
        return jax.jit(
            lambda x, out_row: _scatter_send(
                x, out_row, rows=rows, rows_minor=rows_minor
            )
        )

    # Traced with copy_send_rows returning its index and count operands.
    with monkeypatch.context() as m:
        m.setattr(
            ragged_a2a_moe,
            "copy_send_rows",
            lambda x, index, count, rows: (index, count),
        )
        index_default, index_rows_minor = compiled(False), compiled(True)
        outs = []
        for routing in ("random", "skewed", "dense", "unrouted", "none"):
            need = _need(rng, num_tokens, num_members, routing)
            out_row = jnp.asarray(_send_rows(need, rows, layout))
            outs.append(
                (out_row, index_default(x, out_row), index_rows_minor(x, out_row))
            )
    send_default, send_rows_minor = compiled(False), compiled(True)
    for out_row, want, got in outs:
        np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
        np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))
        count = int(np.asarray(want[1])[0])
        sent = np.asarray(send_rows_minor(x, out_row)).view(np.uint16)
        sent_default = np.asarray(send_default(x, out_row)).view(np.uint16)
        np.testing.assert_array_equal(sent[:count], sent_default[:count])
