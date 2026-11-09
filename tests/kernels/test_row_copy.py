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


def _canonical(bits):
    """The one-hot permute's bf16 bit patterns: +0 for exponent fields 0 and
    255 (zeros, subnormals, infinities, NaNs), every other pattern kept."""
    exponent = bits & 0x7F80
    return np.where((exponent != 0) & (exponent != 0x7F80), bits, 0).astype(np.uint16)


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
@pytest.mark.parametrize("canonical", [True, False])
def test_gather_rows_matches_the_onehot_permute(num_routes, count, hidden, canonical):
    """Every row the expert kernel reads (the computed routes, padded to whole
    16-row blocks with zeros) is bitwise the one-hot permute's row; without
    ``canonical``, from sources that already hold the permute's values."""
    _require_tpu()
    count = num_routes if count < 0 else count
    rng = np.random.default_rng(num_routes * 10 + count)
    # Receive rows as served: 184 at the 2-member proxy (1840 routes), 480 in
    # the 4-host capacity exchange.
    num_src = 184 if num_routes == 1840 else 480
    rows2d = _rows(rng, num_src, hidden)
    src = rows2d
    if not canonical:
        bits = _canonical(np.asarray(rows2d).view(np.uint16))
        src = jnp.asarray(bits.view(jnp.bfloat16))
    index = rng.integers(0, num_src, size=num_routes).astype(np.int32)
    # Routes past the computed ones read no row (the plan's out-of-range index).
    index[count:] = num_src
    sizes = _group_sizes(rng, count)
    # The rows the routes read lie below the filled bound, as in the exchange.
    filled = int(index[:count].max()) + 1 if count else 0
    got = gather_rows(
        src.reshape(num_src, 16, hidden // 16),
        jnp.asarray(index),
        jnp.asarray(sizes),
        jnp.asarray([filled], jnp.int32),
        canonical=canonical,
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
@pytest.mark.parametrize("canonical", [False, True])
def test_copy_send_rows_writes_the_send_rows(
    num_members, num_tokens, routing, canonical
):
    """The send buffer rows the exchange sends are exact copies of their
    token rows (``canonical``: the one-hot permute's values of them), in the
    XLA path's layout."""
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
    got = _scatter_send(x, jnp.asarray(out_row), rows=rows, canonical=canonical)
    # The layout of the XLA path's send buffer, with the token rows copied
    # bit for bit.
    want_shape = jax.eval_shape(lambda x, i: _tile_rows(x[i]), x, jnp.asarray(index))
    assert got.shape == want_shape.shape
    want = np.asarray(x).view(np.uint16)[index].reshape(want_shape.shape)
    if canonical:
        want = _canonical(want)
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16)[:total], want[:total])


@pytest.mark.parametrize("hidden", [8192, 2048])
@pytest.mark.parametrize("cut", [0, 5])
def test_canonical_send_rows_gather_to_the_same_rows(hidden, cut):
    """Every bf16 bit pattern: token rows the send kernel maps once and the
    row gather copies are bitwise the rows the row gather maps from exact
    send copies, the one-hot permute's rows (the ragged all-to-all moves
    whole send rows unchanged into the receive buffer in between)."""
    _require_tpu()
    rng = np.random.default_rng(hidden + cut)
    num_members, topk = 4, 3
    # Two shuffled copies of the 65536 patterns, H per token row.
    num_tokens = 2 * 65536 // hidden
    bits = np.concatenate([rng.permutation(65536), rng.permutation(65536)])
    bits = bits.astype(np.uint16).reshape(num_tokens, hidden)
    x = jnp.asarray(bits.view(jnp.bfloat16))
    # Every token goes to one to three members.
    routed = rng.integers(0, num_members + 1, size=(num_tokens, topk))
    routed[:, 0] = np.arange(num_tokens) % num_members
    need = (routed[:, :, None] == np.arange(num_members)).any(axis=1)
    rows = max(16, -(-int(need.sum()) // 32) * 32)
    out_row, send_index, total = _send_layout(routed, num_members, rows)
    exact = _scatter_send(x, jnp.asarray(out_row), rows=rows)
    mapped = _scatter_send(x, jnp.asarray(out_row), rows=rows, canonical=True)
    sent = np.asarray(mapped).view(np.uint16)[:total].reshape(total, hidden)
    np.testing.assert_array_equal(sent, _canonical(bits[send_index[:total]]))
    # Routes read every received row once, then random ones; ``cut`` leaves
    # the last 16-row block part empty.
    num_routes = -(-(total + 40) // 16) * 16
    index = np.concatenate(
        [rng.permutation(total), rng.integers(0, total, size=num_routes - total)]
    ).astype(np.int32)
    count = num_routes - cut
    index[count:] = rows
    args = (
        jnp.asarray(index),
        jnp.asarray(_group_sizes(rng, count)),
        jnp.asarray([total], jnp.int32),
    )
    want = gather_rows(exact, *args)
    got = gather_rows(mapped, *args, canonical=False)
    written = -(-count // 16) * 16
    got = np.asarray(got).view(np.uint16)[:written]
    np.testing.assert_array_equal(got, np.asarray(want).view(np.uint16)[:written])
    ref = np.zeros((written, hidden), np.uint16)
    ref[:count] = _canonical(bits[send_index[index[:count]]])
    np.testing.assert_array_equal(got, ref)
