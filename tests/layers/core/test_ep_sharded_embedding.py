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
"""EP-sharded embedding lookup against the vocab-parallel (TP-only) lookup."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core.ep_sharded_embedding import (
    DP_AXIS,
    TP_AXIS,
    ep_embedding_lookup,
)

HIDDEN = 256


def _mesh(dp_size, tp_size):
    devices = jax.devices()
    if len(devices) < dp_size * tp_size:
        pytest.skip(f"needs {dp_size * tp_size} devices")
    grid = np.asarray(devices[: dp_size * tp_size]).reshape(dp_size, tp_size)
    return Mesh(grid, (DP_AXIS, TP_AXIS))


def _vocab_parallel(ids, table, mesh):
    """The TP-only lookup: each engine splits the whole table over its TP
    ranks, masks ids outside the local rows, fills them with zeros and
    all-reduces over TP."""
    dp_size, tp_size = mesh.shape[DP_AXIS], mesh.shape[TP_AXIS]
    rows = table.shape[0] // tp_size
    shards = table.reshape(1, tp_size, rows, -1)
    shards = jnp.broadcast_to(shards, (dp_size, *shards.shape[1:]))
    spec = P(DP_AXIS, TP_AXIS)

    def body(i, t):
        i, t = i[0, 0], t[0, 0]
        local = i - jax.lax.axis_index(TP_AXIS) * rows
        owned = (local >= 0) & (local < rows)
        out = jnp.take(t, jnp.where(owned, local, 0), axis=0)
        out = jnp.where(owned[:, None], out, jnp.zeros((), t.dtype))
        if tp_size > 1:
            out = jax.lax.psum(out, TP_AXIS)
        return out[None, None]

    return jax.shard_map(
        body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
    )(ids, shards)


def _ep_sharded(ids, table, mesh):
    dp_size, tp_size = mesh.shape[DP_AXIS], mesh.shape[TP_AXIS]
    rows = table.shape[0] // (dp_size * tp_size)
    shards = table.reshape(dp_size, tp_size, rows, -1)
    starts = (jnp.arange(dp_size * tp_size, dtype=jnp.int32) * rows).reshape(
        dp_size, tp_size, 1
    )
    return jax.jit(lambda i, t, s: ep_embedding_lookup(i, t, s, mesh=mesh))(
        ids, shards, starts
    )


def _table(vocab, dtype, seed):
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((vocab, HIDDEN)).astype(np.float32)
    # Signed zeros and exact powers of two: the sums with +0.0 must agree.
    table[3, :8] = -0.0
    table[vocab - 1, :8] = 2.0**-20
    return jnp.asarray(table).astype(dtype)


def _ids(dp_size, tp_size, tokens, vocab, ep_rows, seed):
    rng = np.random.default_rng(seed + 1)
    per_engine = rng.integers(0, vocab, size=(dp_size, tokens)).astype(np.int32)
    # Shard edges, the signed-zero row, the last row, and repeats.
    edges = [0, 3, ep_rows - 1, ep_rows, 2 * ep_rows - 1, vocab - 1]
    for d in range(dp_size):
        for k, e in enumerate(edges[: min(tokens, len(edges))]):
            per_engine[d, (k * 7 + d) % tokens] = e
    return jnp.asarray(np.broadcast_to(per_engine[:, None], (dp_size, tp_size, tokens)))


def _bits(x):
    x = np.asarray(x)
    view = {2: np.uint16, 4: np.uint32}[x.dtype.itemsize]
    return x.view(view)


@pytest.mark.cpu_test
# TP > 1 only: like the vocab-parallel all-reduce, the DP reduce-scatter adds
# +0.0 to every row, which a TP=1 vocab-parallel lookup (no all-reduce) does not.
@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (4, 2), (1, 8)])
@pytest.mark.parametrize("tokens", [1, 23, 92])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_ep_sharded_lookup_is_bitwise_the_vocab_parallel_lookup(
    dp_size, tp_size, tokens, dtype
):
    mesh = _mesh(dp_size, tp_size)
    ep_rows = 24
    vocab = ep_rows * dp_size * tp_size
    table = _table(vocab, dtype, seed=dp_size * 10 + tp_size)
    ids = _ids(dp_size, tp_size, tokens, vocab, ep_rows, seed=tokens)
    want = _vocab_parallel(ids, table, mesh)
    got = _ep_sharded(ids, table, mesh)
    assert got.shape == (dp_size, tp_size, tokens, HIDDEN)
    np.testing.assert_array_equal(_bits(got), _bits(want))
    # Every engine reads its own tokens' rows, on each of its TP ranks.
    np.testing.assert_array_equal(
        np.asarray(got[:, 0], np.float32), np.asarray(table[ids[:, 0]], np.float32)
    )


@pytest.mark.cpu_test
def test_engines_with_different_ids_get_their_own_rows():
    mesh = _mesh(2, 4)
    ep_rows = 8
    vocab = ep_rows * 8
    table = _table(vocab, jnp.bfloat16, seed=7)
    per_engine = np.stack([np.arange(5), vocab - 1 - np.arange(5)]).astype(np.int32)
    ids = jnp.asarray(np.broadcast_to(per_engine[:, None], (2, 4, 5)))
    got = np.asarray(_ep_sharded(ids, table, mesh), np.float32)
    for d in range(2):
        for t in range(4):
            np.testing.assert_array_equal(
                got[d, t], np.asarray(table[per_engine[d]], np.float32)
            )
