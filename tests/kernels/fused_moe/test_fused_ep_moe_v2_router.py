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
"""Exact selector parity across row tiles, tie rules and separate weight tables.

CPU cases execute the Pallas interpreter; TPU cases compile the same inputs
through Mosaic on one device. Neither mode runs the full MoE layer.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import router_ops

# Wide enough that a top-k spans several 128-lane blocks at the shapes the
# router actually sees, small enough to check exhaustively against numpy.
EXPERTS = 256
TOPK = 8


@pytest.fixture(
    params=[
        pytest.param(True, marks=pytest.mark.cpu_test, id="cpu"),
        pytest.param(False, id="tpu"),
    ]
)
def interpret(request: pytest.FixtureRequest) -> bool:
    if not request.param and jax.default_backend() != "tpu":
        pytest.skip("Native Pallas lowering requires a TPU")
    return request.param


def _select(
    scores: np.ndarray,
    weight_scores: np.ndarray | None = None,
    topk: int = TOPK,
    block_rows: int | None = None,
    *,
    interpret: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows = scores.shape[0]
    select = jax.jit(
        functools.partial(
            router_ops.pallas_select,
            topk=topk,
            block_rows=rows if block_rows is None else block_rows,
            interpret=interpret,
        )
    )
    w, i = select(
        jnp.asarray(scores),
        weight_scores=None if weight_scores is None else jnp.asarray(weight_scores),
    )
    assert w.shape == i.shape == (rows, topk)
    assert w.dtype == jnp.float32
    assert i.dtype == jnp.int32
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


def _scores(rows, seed, scale=1.0, experts=EXPERTS):
    rng = np.random.default_rng(seed)
    return (rng.random((rows, experts), dtype=np.float32) * scale).astype(np.float32)


@pytest.mark.parametrize(
    "rows,experts,topk,block_rows",
    [
        (8, 256, 8, 8),
        (256, 256, 8, 256),
        (512, 256, 8, 256),
        (1024, 256, 8, 256),
        (256, 384, 6, 256),
        pytest.param(2048, 384, 6, 256, id="tuned-2048-rows"),
        pytest.param(512, 384, 6, 64, id="align-partial-tiles"),
        pytest.param(192, 384, 6, 64, id="fallback-to-full-rows"),
        (256, 384, 6, 128),
        pytest.param(128, 128, 1, 128, id="topk-one"),
        pytest.param(8, 16, 16, 8, id="topk-all-experts"),
    ],
)
@pytest.mark.parametrize("separate_weights", [False, True])
def test_selection_matches_reference(
    rows, experts, topk, block_rows, separate_weights, interpret
):
    """Transposition and tiling preserve every token's expert IDs and weights."""
    scores = _scores(rows, seed=rows, experts=experts)
    weights = (
        _scores(rows, seed=rows + 1, experts=experts) if separate_weights else None
    )
    w, i = _select(scores, weights, topk, block_rows, interpret=interpret)
    ref_w, ref_i = _reference(scores, scores if weights is None else weights, topk)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


def test_the_bias_moves_the_selection_and_not_the_weights(interpret):
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

    w, i = _select(biased, raw, interpret=interpret)
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


@pytest.mark.parametrize("separate_weights", [False, True])
@pytest.mark.parametrize("partial_nan", [False, True], ids=["ties", "partial-nan"])
def test_ties_choose_lowest_expert_across_lane_boundaries(
    separate_weights, partial_nan, interpret
):
    scores = np.zeros((256, 384), dtype=np.float32)
    scores[:, [127, 128, 255, 256, 382, 383]] = 1.0
    if partial_nan:
        scores[:, :127] = np.nan
    weights = _scores(256, seed=11, experts=384) if separate_weights else None
    w, i = _select(scores, weights, topk=6, block_rows=128, interpret=interpret)
    ref_w, ref_i = _reference(
        np.where(np.isnan(scores), -3.0e38, scores),
        scores if weights is None else weights,
        topk=6,
    )
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


# Adapt the every-expert and tiny-score cases from vllm-project/vllm-torchtpu
# PR #2480 to this selector's unnormalized, multi-tile output contract.
@pytest.mark.parametrize("separate_weights", [False, True])
def test_every_expert_id_preserves_token_and_topk_position(separate_weights, interpret):
    """Each output slot visits every expert, with distinct weights per token."""
    rows, experts, topk = 512, 384, 6
    expected_ids = (
        np.arange(rows)[:, None] + np.arange(topk)[None, :] * (experts // topk)
    ) % experts
    for slot in range(topk):
        np.testing.assert_array_equal(
            np.unique(expected_ids[:, slot]), np.arange(experts)
        )
    scores = np.full((rows, experts), -1, dtype=np.float32)
    np.put_along_axis(scores, expected_ids, np.arange(topk, 0, -1)[None, :], axis=1)
    weights = (
        np.arange(rows * experts, dtype=np.float32).reshape(rows, experts)
        if separate_weights
        else None
    )
    w, i = _select(scores, weights, topk, block_rows=128, interpret=interpret)
    np.testing.assert_array_equal(i, expected_ids)
    np.testing.assert_array_equal(
        w,
        np.take_along_axis(
            scores if weights is None else weights, expected_ids, axis=1
        ),
    )


@pytest.mark.parametrize("separate_weights", [False, True])
@pytest.mark.parametrize("case", ["zeros", "uniform", "tiny"])
def test_zero_uniform_and_tiny_scores_preserve_selection(
    case, separate_weights, interpret
):
    """Zero/tied scores select distinct lowest IDs; tiny weights stay unnormalized."""
    scores = _scores(256, seed=17, scale=1e-22, experts=384)
    if case == "zeros":
        scores.fill(0)
    elif case == "uniform":
        scores.fill(1 / 384)
    weights = (
        _scores(256, seed=18, scale=1e-22, experts=384) if separate_weights else None
    )
    w, i = _select(scores, weights, topk=6, block_rows=128, interpret=interpret)
    ref_w, ref_i = _reference(scores, scores if weights is None else weights, topk=6)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


@pytest.mark.parametrize("separate_weights", [False, True])
def test_all_nan_rows_preserve_sentinel_and_repeated_expert_zero(
    separate_weights, interpret
):
    """The fused router intentionally leaves invalid-row masking to its caller."""
    scores = np.full((8, 384), np.nan, dtype=np.float32)
    weights = _scores(8, seed=12, experts=384) if separate_weights else None
    w, i = _select(scores, weights, topk=6, interpret=interpret)
    np.testing.assert_array_equal(i, np.zeros((8, 6), dtype=np.int32))
    expected = (
        np.full((8, 6), -3.0e38, dtype=np.float32)
        if weights is None
        else np.broadcast_to(weights[:, :1], (8, 6))
    )
    np.testing.assert_array_equal(w, expected)


@pytest.mark.cpu_test
@pytest.mark.parametrize("weight_shape", [(128, 383), (127, 384)])
def test_rejects_mismatched_weight_table(weight_shape):
    with pytest.raises(ValueError, match="must match scores"):
        _select(
            np.zeros((128, 384), dtype=np.float32),
            np.zeros(weight_shape, dtype=np.float32),
            interpret=True,
        )


@pytest.mark.cpu_test
def test_rejects_nondivisible_requested_row_tile():
    with pytest.raises(AssertionError, match=r"\(257, 256\)"):
        _select(
            np.zeros((257, 384), dtype=np.float32),
            block_rows=256,
            interpret=True,
        )
