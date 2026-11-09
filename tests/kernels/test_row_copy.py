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
    got = _scatter_send(x, jnp.asarray(out_row.T), rows=rows)
    # The layout of the XLA path's send buffer, with the token rows copied
    # bit for bit.
    want_shape = jax.eval_shape(lambda x, i: _tile_rows(x[i]), x, jnp.asarray(index))
    assert got.shape == want_shape.shape
    want = np.asarray(x).view(np.uint16)[index].reshape(want_shape.shape)
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16)[:total], want[:total])


def _token_major_index(out_row, rows):
    """The token of each send row (0 where none) from the send rows [S, D],
    by one compare-and-sum over [rows, S, D], and the filled row count."""
    token = np.arange(out_row.shape[0], dtype=np.int32)[:, None]
    hit = out_row[None] == np.arange(rows)[:, None, None]
    index = np.where(hit, token[None], 0).sum(axis=(1, 2), dtype=np.int32)
    return index, np.where(out_row < rows, out_row + 1, 0).max(initial=0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_members", [2, 8, 16, 32])
@pytest.mark.parametrize("layout", ["capacity", "overflow", "regions"])
def test_scatter_send_index_matches_the_token_major_build(
    monkeypatch, num_members, layout
):
    """The index and count `_scatter_send` hands `copy_send_rows` equal the
    token-major build's on every row: capacity buffers filled without gaps or
    smaller than the step, and the full-size exchange's region per member,
    with empty and full destination groups."""
    seen = {}

    def _copy_send_rows(x, index, count, rows):
        seen.update(index=np.asarray(index), count=int(count[0]))

    monkeypatch.setattr(ragged_a2a_moe, "copy_send_rows", _copy_send_rows)
    rng = np.random.default_rng(num_members)
    for num_tokens in (12, 112):
        for density in (0.0, 0.05, 0.3, 1.0):
            need = rng.random((num_tokens, num_members)) < density
            if layout == "regions":
                rows = num_members * num_tokens
                start = np.arange(num_members) * num_tokens
            else:
                sizes = need.sum(axis=0)
                start = np.cumsum(sizes) - sizes
                rows = max(16, -(-int(sizes.sum()) // 32) * 32)
                if layout == "overflow":
                    rows = max(16, rows // 2 // 16 * 16)
            out_row = np.where(need, start + np.cumsum(need, axis=0) - 1, rows)
            out_row = out_row.astype(np.int32)
            x = jnp.zeros((num_tokens, 128), jnp.bfloat16)
            _scatter_send(x, jnp.asarray(out_row.T), rows=rows)
            index, count = _token_major_index(out_row, rows)
            np.testing.assert_array_equal(seen["index"], index)
            assert seen["count"] == count
