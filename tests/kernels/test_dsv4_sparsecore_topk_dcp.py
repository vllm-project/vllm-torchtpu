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
"""Tests for `sparsecore_topk_dcp`, the top-k over rows split across DCP ranks.

The contract is exact and deterministic, so the reference is too: the union of
the ranks' outputs is the top-k of the whole row, with ties at the k-th score
taken in (rank, column) order. Rows are drawn so that each path of the kernel
runs: several rows per subcore, one, padding rows no rank sees, and parts
longer than tile memory, which are streamed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk_dcp import (
    ROW_WORDS,
    sparsecore_topk_dcp,
)

pytestmark = pytest.mark.tpu_test

P = jax.sharding.PartitionSpec
AXIS = "dcp"


def _need_devices(dcp_size):
    if len(jax.devices()) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(jax.devices())}")


def _run(scores, lengths, k, lanes=False, **kwargs):
    """scores f32[dcp, b, n], lengths i32[dcp, b] -> i32[dcp, b, k].

    `lanes` puts a leading size-1 axis on the mesh, as the served DCP mesh
    carries for the TP lanes DCP does not shard over.
    """
    dcp_size = scores.shape[0]
    devices = np.array(jax.devices()[:dcp_size])
    if lanes:
        mesh = jax.sharding.Mesh(devices.reshape(1, dcp_size), ("lane", AXIS))
    else:
        mesh = jax.sharding.Mesh(devices, (AXIS,))

    def local(s, ln, bounds):
        return sparsecore_topk_dcp(
            s[0], k, ln[0], bounds, dcp_size=dcp_size, axis_name=AXIS, **kwargs
        )[None]

    fn = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS), P()),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    bounds = lengths.max(axis=0)
    return np.asarray(
        fn(jnp.asarray(scores), jnp.asarray(lengths), jnp.asarray(bounds))
    )


def _expected(scores, lengths, k):
    """The ranks' expected column sets: top-k, ties in (rank, column) order."""
    dcp_size, b, _ = scores.shape
    want = []
    for row in range(b):
        ranks = np.concatenate(
            [np.full(lengths[r, row], r) for r in range(dcp_size)]
        ).astype(np.int64)
        cols = np.concatenate([np.arange(lengths[r, row]) for r in range(dcp_size)])
        vals = np.concatenate(
            [scores[r, row, : lengths[r, row]] for r in range(dcp_size)]
        )
        keep = vals > -np.inf
        ranks, cols, vals = ranks[keep], cols[keep], vals[keep]
        top = np.lexsort((cols, ranks, -vals))[:k]
        want.append([set(cols[top][ranks[top] == r].tolist()) for r in range(dcp_size)])
    return want


def _check(out, scores, lengths, k):
    dcp_size, b, _ = scores.shape
    for row, sets in enumerate(_expected(scores, lengths, k)):
        for r in range(dcp_size):
            got = out[r, row]
            cols = got[got >= 0]
            assert (got[cols.size :] == -1).all(), f"row {row} rank {r}: not packed"
            assert set(cols.tolist()) == sets[r], (
                f"row {row} rank {r}: {len(sets[r] - set(cols.tolist()))} missing,"
                f" {len(set(cols.tolist()) - sets[r])} spurious"
            )


def _case(dcp_size, b, n, lo, hi, seed, levels=0, live=None):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(lo, hi + 1, size=(dcp_size, b)).astype(np.int32)
    if live is not None:
        lengths[:, live:] = 0
    if levels:
        # Few distinct values: the k-th score is shared by many entries.
        scores = rng.integers(0, levels, size=(dcp_size, b, n)) / levels
    else:
        scores = rng.standard_normal((dcp_size, b, n))
    scores = scores.astype(np.float32)
    scores[rng.random(scores.shape) < 0.01] = -np.inf
    return scores, np.minimum(lengths, n)


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize(
    "b,n,lo,hi,levels,live",
    [
        (64, 4096, 0, 4096, 0, None),  # many rows per subcore, some empty
        (64, 4096, 1000, 4096, 16, None),  # ties at the k-th score
        (96, 2048, 0, 300, 0, 40),  # under k entries; padding rows
        (16, 16384, 8000, 16384, 0, None),  # fewer rows than subcores
    ],
)
def test_matches_the_global_topk(dcp_size, b, n, lo, hi, levels, live):
    _need_devices(dcp_size)
    k = 512
    scores, lengths = _case(dcp_size, b, n, lo, hi, dcp_size, levels, live)
    _check(_run(scores, lengths, k), scores, lengths, k)


@pytest.mark.multichip
@pytest.mark.parametrize("b", [16, 64])
def test_mesh_with_a_lane_axis(b):
    """The served DCP mesh is (lane, dcp): peers are addressed on `AXIS` only."""
    dcp_size = 4
    _need_devices(dcp_size)
    scores, lengths = _case(dcp_size, b, 4096, 0, 4096, 9)
    _check(_run(scores, lengths, 512, lanes=True), scores, lengths, 512)


@pytest.mark.multichip
def test_parts_longer_than_tile_memory_are_streamed():
    dcp_size = 2
    _need_devices(dcp_size)
    n = 2 * ROW_WORDS + 4096
    scores, lengths = _case(dcp_size, 4, n, ROW_WORDS + 1, n, 3)
    _check(_run(scores, lengths, 2048), scores, lengths, 2048)


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 8])
@pytest.mark.parametrize(
    "variant",
    [{}, {"one_round": True}, {"pack16": True}],
    ids=["default", "one_round", "pack16"],
)
@pytest.mark.parametrize(
    "b,levels",
    [
        (16, 0),  # one row per subcore: gated one-round exchange
        (64, 0),  # two rows per subcore
        (256, 0),  # owner exchange; streamed survivors compacted
        (256, 1),  # constant rows overflow compaction and stream
    ],
)
def test_matches_the_two_round_streaming_kernel(dcp_size, variant, b, levels):
    """Every switch must give the old path's output exactly, order included."""
    _need_devices(dcp_size)
    k = 2048
    n = 40960
    scores, lengths = _case(dcp_size, b, n, 24576, n, 11, levels)
    new = _run(scores, lengths, k, **variant)
    old = _run(scores, lengths, k, one_round=False, no_restream=False)
    np.testing.assert_array_equal(new, old)
    _check(new, scores, lengths, k)


@pytest.mark.multichip
def test_comparison_detects_a_wrong_index():
    """The check above must be able to fail."""
    dcp_size = 2
    _need_devices(dcp_size)
    k = 512
    scores, lengths = _case(dcp_size, 32, 2048, 1000, 2048, 5)
    out = _run(scores, lengths, k)
    _check(out, scores, lengths, k)
    bad = out.copy()
    row = bad[0, 0]
    unused = sorted(set(range(lengths[0, 0])) - set(row[row >= 0].tolist()))
    row[0] = unused[0]
    with pytest.raises(AssertionError):
        _check(bad, scores, lengths, k)
