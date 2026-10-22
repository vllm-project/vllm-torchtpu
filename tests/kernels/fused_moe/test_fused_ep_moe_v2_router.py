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
lowering: `tests/kernels/fused_moe/test_fused_ep_moe_v2.py` runs the full layer
on eight TPUs. The decode tile regression below also checks Mosaic lowering
when a TPU is available.
"""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import router_ops  # noqa: E402

# Wide enough that a top-k spans several 128-lane blocks at the shapes the
# router actually sees, small enough to check exhaustively against numpy.
EXPERTS = 256
TOPK = 8


def _select(scores, weight_scores=None, topk=TOPK, block_rows=None, interpret=True):
    """Run selection and return host arrays."""
    rows = scores.shape[0]
    w, i = router_ops.pallas_select(
        jnp.asarray(scores),
        topk=topk,
        block_rows=rows if block_rows is None else block_rows,
        weight_scores=None if weight_scores is None else jnp.asarray(weight_scores),
        interpret=interpret,
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


@pytest.mark.parametrize("rows,block_rows", [(32, 32), (288, 288), (576, 288)])
@pytest.mark.parametrize("separate_weights", [False, True])
@pytest.mark.parametrize("map_dtype", [jnp.int32, jnp.uint32])
def test_fused_expert_map_preserves_logical_ties_and_weight_bits(
    rows, block_rows, separate_weights, map_dtype
):
    rng = np.random.default_rng(rows)
    scores = rng.integers(0, 4, (rows, 384)).astype(np.float32)
    scores[0], scores[1], scores[2] = np.nan, -np.inf, np.inf
    weights = rng.normal(size=scores.shape).astype(np.float32)
    weights[3] = np.copysign(np.zeros(384, np.float32), np.arange(384) % 2 - 0.5)
    weight_scores = jnp.asarray(weights) if separate_weights else None
    # Reverse physical IDs so a physical-index tie breaker would be wrong.
    mapping = jnp.arange(383, -1, -1, dtype=map_dtype)
    kwargs = dict(topk=6, block_rows=block_rows, weight_scores=weight_scores)
    ref_w, logical_ids = router_ops.pallas_select(
        jnp.asarray(scores), interpret=True, **kwargs
    )
    actual_w, physical_ids = router_ops.pallas_select(
        jnp.asarray(scores), expert_id_map=mapping, interpret=True, **kwargs
    )
    np.testing.assert_array_equal(physical_ids, np.asarray(mapping)[logical_ids])
    np.testing.assert_array_equal(
        np.asarray(actual_w).view(np.uint32), np.asarray(ref_w).view(np.uint32)
    )


@pytest.mark.parametrize(
    "mapping", [np.zeros(383, np.int32), np.zeros(384, np.float32)]
)
def test_fused_expert_map_rejects_invalid_shape_or_dtype(mapping):
    with pytest.raises(ValueError, match="one integer per expert"):
        router_ops.pallas_select(
            jnp.zeros((32, 384)),
            topk=6,
            block_rows=32,
            expert_id_map=jnp.asarray(mapping),
            interpret=True,
        )


def test_fused_expert_map_dispatch_keeps_unmeasured_shapes(monkeypatch):
    from vllm_torchtpu.kernels.fused_moe.v2 import tuned_block_sizes as tuning

    monkeypatch.setattr(tuning, "logger", Mock())
    key = tuning.RouterTuningKey("TPU7x", 288, 384, 6, True)
    assert tuning.use_fused_expert_id_map(key)
    for changed in (
        replace(key, rows=2048),
        replace(key, separate_weights=False),
        replace(key, device_kind="TPU v6e"),
        replace(key, experts=256),
        replace(key, topk=8),
    ):
        assert not tuning.use_fused_expert_id_map(changed)


@pytest.mark.parametrize("separate_weights", [False, True])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("interpret", [True, False])
def test_decode_selector_whole_batch_matches_tiled(separate_weights, ties, interpret):
    """The non-power-of-two tile preserves routing and noaux weights."""
    if not interpret and jax.default_backend() != "tpu":
        pytest.skip("Compiled selector validation requires TPU")
    rng = np.random.default_rng(1024)
    scores = rng.random((288, 384), dtype=np.float32)
    if ties:
        scores = np.floor(scores * 4)
    scores[0] = np.nan
    scores[1] = -np.inf
    weights = rng.random(scores.shape, dtype=np.float32) if separate_weights else None
    actual_w, actual_i = _select(
        scores, weights, topk=6, block_rows=288, interpret=interpret
    )
    old_w, old_i = _select(scores, weights, topk=6, block_rows=32, interpret=interpret)
    ref_w, ref_i = _reference(
        scores[2:], scores[2:] if weights is None else weights[2:], topk=6
    )
    np.testing.assert_array_equal(actual_i, old_i)
    np.testing.assert_array_equal(actual_i[2:], ref_i)
    np.testing.assert_array_equal(actual_w, old_w)
    np.testing.assert_array_equal(actual_w[2:], ref_w)


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
