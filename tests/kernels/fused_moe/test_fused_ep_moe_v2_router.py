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
"""`pallas_select`'s two-table mode: select by one score table, weigh by another.

DeepSeek-V4's `noaux_tc` decides the top-k on scores that carry a per-expert
correction bias and then weighs the chosen experts by the scores WITHOUT it.
Folding the bias into the logits cannot express that, so the selector takes a
second [R, N] table and reads the selected weights from it.

The failure mode this file is built around is that the two tables come apart
silently. Selecting correctly and weighing from the wrong table is not an
error at any layer below -- every index is a real expert, every weight is a
plausible float, and the renormalization downstream rescales whatever it is
handed. What it produces is a model that routes right and weighs wrong, which
has no symptom other than quality. So the assertions here are equalities
against a numpy gather, not tolerances, and the bias in the fixtures is large
enough to REORDER the selection rather than perturb it -- a test whose bias
changed nothing would pass against a kernel that ignored the bias entirely.

`interpret=True` runs the kernel body on the host, which is what lets the
selection rule be gated without a chip, the same way `tests/kernels/
test_router_topk.py` gates the sibling selector. It does not gate Mosaic
lowering: `tests/kernels/fused_moe/test_fused_ep_moe_v2.py` is what runs this
code on a TPU, and it needs eight of them.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import router_ops  # noqa: E402

# Wide enough that a top-k spans several 128-lane blocks at the shapes the
# router actually sees, small enough to check exhaustively against numpy.
EXPERTS = 256
TOPK = 8


def _select(scores, weight_scores=None, topk=TOPK, block_rows=None):
    """The kernel on the host, as numpy."""
    rows = scores.shape[0]
    w, i = router_ops.pallas_select(
        jnp.asarray(scores),
        topk=topk,
        block_rows=rows if block_rows is None else block_rows,
        weight_scores=None if weight_scores is None else jnp.asarray(weight_scores),
        interpret=True,
    )
    return np.asarray(w), np.asarray(i)


def _reference(scores, weight_scores, topk=TOPK):
    """Top-k by `scores`, weights gathered from `weight_scores`.

    Ties go to the lowest column, which is the kernel's rule: it takes the
    row max and then the lowest column equal to it. `argsort(-x, stable)`
    is the same rule, and `np.argpartition` is NOT -- it is unordered within
    the partition, so it disagrees on exactly the rows that matter.
    """
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :topk]
    return np.take_along_axis(weight_scores, idx, axis=1), idx


def _scores(rows, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.random((rows, EXPERTS), dtype=np.float32) * scale).astype(np.float32)


@pytest.mark.parametrize(
    "rows,block_rows", [(8, 8), (256, 256), (512, 256), (1024, 256)]
)
def test_one_table_mode_is_unchanged(rows, block_rows):
    """The default path still returns the selected scores as the weights.

    The two-table mode was added by threading an optional ref through the
    same selection loop, so the one-table path is reachable code that no
    longer has a test of its own anywhere off-chip.
    """
    scores = _scores(rows, seed=rows)
    w, i = _select(scores, block_rows=block_rows)
    ref_w, ref_i = _reference(scores, scores)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


@pytest.mark.parametrize(
    "rows,block_rows", [(8, 8), (256, 256), (512, 256), (1024, 256)]
)
def test_weights_come_from_the_second_table(rows, block_rows):
    """Indices from `scores`, weights from `weight_scores`, exactly."""
    scores = _scores(rows, seed=rows)
    weights = _scores(rows, seed=rows + 1)
    w, i = _select(scores, weights, block_rows=block_rows)
    ref_w, ref_i = _reference(scores, weights)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


def test_a_row_with_fewer_finite_scores_than_topk_weighs_finite():
    """The weight table is NaN-flushed too, not just the selection table.

    A row with fewer finite scores than `topk` runs the selector out of real
    columns, and every slot past that point falls to the sentinel tie's lowest
    column -- a NaN column here. In two-table mode that column's WEIGHT is
    what comes back, and `layer.py` then divides the row by a sum containing
    it, so one NaN weight makes the whole row NaN. The all-finite mask one
    layer up does not fire: this row has finite scores.
    """
    scores = np.full((8, EXPERTS), np.nan, dtype=np.float32)
    weights = np.full((8, EXPERTS), np.nan, dtype=np.float32)
    scores[:, 1:3] = np.array([0.5, 0.25], dtype=np.float32)
    weights[:, 1:3] = np.array([0.731, 0.622], dtype=np.float32)

    w, i = _select(scores, weights)
    assert np.isfinite(w).all(), "a NaN weight escaped the selector"
    # The two finite columns are picked in order and weighed from the second
    # table; the slots with nothing left to pick carry the sentinel.
    np.testing.assert_array_equal(i[:, :2], np.tile([1, 2], (8, 1)))
    np.testing.assert_allclose(w[:, :2],
                               np.tile([0.731, 0.622], (8, 1)),
                               rtol=1e-6)
    assert (w[:, 2:] == router_ops.NEG).all()


def test_a_weight_table_of_a_different_dtype_is_refused_here():
    """A bf16 table stores into the f32 weights ref and fails as a Mosaic
    lowering error that names neither this argument nor this call. The gate
    output IS bf16; only an explicit widen on the serving path makes it f32."""
    scores = jnp.asarray(_scores(8, seed=3))
    with pytest.raises(ValueError, match="dtype"):
        router_ops.pallas_select(scores,
                                 topk=TOPK,
                                 block_rows=8,
                                 weight_scores=scores.astype(jnp.bfloat16),
                                 interpret=True)


def test_the_bias_moves_the_selection_and_not_the_weights():
    """The `noaux_tc` call as the layer makes it, end to end.

    The bias is large against the score spread, so the biased top-k is a
    genuinely different expert set from the unbiased one -- asserted here,
    because a fixture where it is not would pass against a kernel that
    dropped the bias on the floor.
    """
    raw = _scores(64, seed=7)
    rng = np.random.default_rng(8)
    bias = (rng.random(EXPERTS, dtype=np.float32) * 4.0).astype(np.float32)
    biased = raw + bias[None, :]

    w, i = _select(biased, raw)
    ref_w, ref_i = _reference(biased, raw)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)

    # The weights are the unbiased scores at the biased picks: neither the
    # biased scores, nor the weights the unbiased selection would have given.
    np.testing.assert_array_equal(w, np.take_along_axis(raw, i, axis=1))
    assert not np.array_equal(w, np.take_along_axis(biased, i, axis=1))
    _, unbiased_i = _reference(raw, raw)
    assert not np.array_equal(i, unbiased_i), (
        "the fixture's bias does not reorder the top-k, so this test cannot "
        "tell a kernel that applies it from one that ignores it"
    )
