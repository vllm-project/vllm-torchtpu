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
"""Numerics gate for the sort-free router top-k, against ``torch.topk``.

The contract: values are bit-identical always; ids are identical wherever the
k-boundary is tie-free; and where ids differ the scores are equal, which is
the only freedom either path has -- torch_tpu emits ``is_stable=false``.
"""

import pytest
import torch

from vllm_torchtpu.layers.common.rowmax_topk import NEG, rowmax_topk

EXPERTS = 512
TOPK = 10


def _reference(scores: torch.Tensor, k: int):
    return torch.topk(scores, k=k, dim=-1)


def _assert_matches_reference(scores: torch.Tensor, k: int = TOPK):
    """Values bit-identical; ids identical except across exact ties."""
    ref_v, ref_i = _reference(scores, k)
    got_v, got_i = rowmax_topk(scores, k)

    assert got_i.dtype == torch.int32
    assert got_v.dtype == scores.dtype
    assert torch.equal(got_v, ref_v), "selected values differ"

    differ = (got_i != ref_i).any(dim=-1)
    for row in differ.nonzero(as_tuple=True)[0].tolist():
        ours, theirs = set(got_i[row].tolist()), set(ref_i[row].tolist())
        only_ours, only_theirs = ours - theirs, theirs - ours
        # A set difference is only legitimate between equally scored experts.
        our_scores = sorted(scores[row, list(only_ours)].tolist())
        their_scores = sorted(scores[row, list(only_theirs)].tolist())
        assert our_scores == their_scores, (
            f"row {row}: swapped experts have different scores "
            f"{our_scores} vs {their_scores}")
        # And ours must be the lowest-indexed of the tied candidates.
        for e in only_ours:
            tied = [
                int(i)
                for i in (scores[row] == scores[row,
                                                e]).nonzero(as_tuple=True)[0]
            ]
            picked = [i for i in tied if i in ours]
            assert picked and min(picked) == min(tied), (
                f"row {row}: tie at score {scores[row, e]} not resolved to "
                f"the lowest expert id")


def test_descending_and_shapes():
    torch.manual_seed(0)
    scores = torch.rand(37, EXPERTS, dtype=torch.float32)
    v, i = rowmax_topk(scores, TOPK)
    assert v.shape == (37, TOPK) and i.shape == (37, TOPK)
    assert (v[:, :-1] >= v[:, 1:]).all(), "values must be descending"
    assert (i >= 0).all() and (i < EXPERTS).all()
    # Every returned id must index its returned value.
    assert torch.equal(torch.gather(scores, 1, i.long()), v)


@pytest.mark.parametrize("rows", [1, 8, 64, 257])
@pytest.mark.parametrize("k", [1, 2, TOPK])
def test_matches_torch_topk_uniform(rows, k):
    torch.manual_seed(rows * 100 + k)
    _assert_matches_reference(torch.rand(rows, EXPERTS), k)


def test_matches_torch_topk_on_real_routing_distribution():
    """The model's own path: bf16 gate logits -> f32 softmax -> top-k, where
    the 8-bit mantissa makes exact ties common."""
    torch.manual_seed(1234)
    for sigma in (0.5, 1.0, 2.0, 4.0):
        logits = (torch.randn(2048, EXPERTS) * sigma).to(torch.bfloat16)
        scores = logits.float().softmax(dim=-1)
        _assert_matches_reference(scores)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expert_set_identical_where_the_top_k_is_unique(seed):
    """Where the top-k is well posed, the two paths must agree exactly.

    About 11% of realistic rows tie across the k-boundary, and on those rows
    neither path is defined.
    """
    torch.manual_seed(seed)
    logits = torch.randn(4096, EXPERTS).to(torch.bfloat16)
    scores = logits.float().softmax(dim=-1)
    kth = scores.topk(TOPK + 1, dim=-1).values
    unique = kth[:, TOPK - 1] != kth[:, TOPK]
    assert unique.float().mean() > 0.5, "sanity: most rows are unambiguous"

    ours = rowmax_topk(scores, TOPK)[1][unique].sort(dim=-1).values
    ref = torch.topk(scores, TOPK, dim=-1).indices.to(
        torch.int32)[unique].sort(dim=-1).values
    assert torch.equal(ours, ref)


def test_ambiguous_rows_move_at_most_two_ids():
    torch.manual_seed(5)
    scores = torch.randn(4096, EXPERTS).to(torch.bfloat16).float().softmax(-1)
    ours = rowmax_topk(scores, TOPK)[1].long()
    ref = torch.topk(scores, TOPK, dim=-1).indices
    hit_ours = torch.zeros(len(scores), EXPERTS, dtype=torch.bool)
    hit_ref = torch.zeros_like(hit_ours)
    hit_ours.scatter_(1, ours, True)
    hit_ref.scatter_(1, ref, True)
    moved = TOPK - (hit_ours & hit_ref).sum(dim=1)
    assert int(moved.max()) <= 2, "a tie-break may move only the last slots"


def test_all_equal_row_selects_lowest_experts():
    """A padded token: zero hidden state -> uniform scores -> pure tie."""
    scores = torch.full((4, EXPERTS), 1.0 / EXPERTS)
    v, i = rowmax_topk(scores, TOPK)
    assert torch.equal(v, torch.full((4, TOPK), 1.0 / EXPERTS))
    expected = torch.arange(TOPK, dtype=torch.int32).expand(4, TOPK)
    assert torch.equal(i, expected), "all-tie must resolve to experts 0..k-1"


def test_adversarial_ties_straddling_the_boundary():
    """Blocks of equal scores, including one that straddles rank k-1/k."""
    scores = torch.zeros(1, EXPERTS)
    # Five clear winners, then ten equal scores for the remaining five slots.
    scores[0, 100:105] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0])
    scores[0, 200:210] = 4.0
    scores[0, 300:] = 1.0
    _assert_matches_reference(scores)
    _, i = rowmax_topk(scores, TOPK)
    assert i[0, :5].tolist() == [100, 101, 102, 103, 104]
    assert i[0, 5:].tolist() == list(range(200, 205)), \
        "tied block must be taken lowest-id first"


def test_duplicate_of_the_maximum_is_not_collapsed():
    """Two experts holding the max must consume two slots, not one."""
    scores = torch.zeros(1, EXPERTS)
    scores[0, 5] = scores[0, 400] = 1.0
    v, i = rowmax_topk(scores, 3)
    assert v[0].tolist() == [1.0, 1.0, 0.0]
    assert i[0, :2].tolist() == [5, 400]


def test_negative_zero_ties_with_zero():
    scores = torch.zeros(1, EXPERTS)
    scores[0, 3] = -0.0
    _assert_matches_reference(scores, k=2)


def test_negative_scores_are_selected():
    """``e_score_correction_bias`` can push the choice scores negative."""
    torch.manual_seed(7)
    scores = -torch.rand(16, EXPERTS) * 1e3
    _assert_matches_reference(scores)


def test_partial_nan_row_ignores_the_nans():
    scores = torch.rand(1, EXPERTS)
    scores[0, :5] = float("nan")
    v, i = rowmax_topk(scores, TOPK)
    assert not torch.isnan(v).any()
    assert (i >= 5).all(), "NaN columns must never be selected"
    finite = scores[0, 5:]
    assert torch.equal(v[0], torch.topk(finite, TOPK).values)


def test_all_nan_row_stays_poisoned():
    """No real maximum exists: keep NaN rather than silently routing."""
    scores = torch.full((2, EXPERTS), float("nan"))
    v, i = rowmax_topk(scores, TOPK)
    assert torch.isnan(v).all()
    assert (i >= 0).all() and (i < EXPERTS).all(), "ids stay real experts"


def test_all_nan_row_selects_k_distinct_experts():
    """A poisoned row must still name ``k`` *different* experts.

    Every column of a fully-NaN row is the flush sentinel, so a mask value
    equal to that sentinel leaves every column tied with the row max and each
    pass returns column 0: one expert, ``k`` times. The sort path names ``k``
    distinct ones, so the MoE gathers ten times fewer expert bytes on exactly
    the rows a corrupted hidden state produces -- a selector-dependent change
    in how much work the layer does, on top of the poisoned weights.

    Which ``k`` is a free choice (``torch.topk`` leaves the order of an
    all-NaN row unspecified and returns an arbitrary permutation); 0..k-1 is
    the lowest-index rule this module already applies to ties.
    """
    scores = torch.full((3, EXPERTS), float("nan"))
    _, i = rowmax_topk(scores, TOPK)
    for row in i:
        assert sorted(row.tolist()) == list(range(TOPK))
    ref_i = torch.topk(scores, TOPK, dim=-1).indices
    for got, ref in zip(i, ref_i):
        assert len(set(got.tolist())) == len(set(ref.tolist())) == TOPK


def test_partial_nan_row_selects_k_distinct_finite_experts():
    scores = torch.rand(2, EXPERTS)
    scores[:, :5] = float("nan")
    v, i = rowmax_topk(scores, TOPK)
    assert not torch.isnan(v).any()
    for row in i:
        assert len(set(row.tolist())) == TOPK
        assert min(row.tolist()) >= 5, "NaN columns must never be selected"


# Three spellings of one hazard: a row whose maximum is at or below the mask
# value. Only NaN was covered before; ``-inf`` and ``-FLT_MAX`` collapsed the
# same way, and ``-inf`` is reachable through an ``-inf``
# ``e_score_correction_bias``, the standard idiom for masking an expert out.
SENTINELS = pytest.mark.parametrize("bad", [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("-inf"), id="neg_inf"),
    pytest.param(torch.finfo(torch.float32).min, id="neg_flt_max"),
])


@SENTINELS
def test_row_at_or_below_the_sentinel_selects_k_distinct_experts(bad):
    """The mask must never outrank a real column. See ``_select_kernel``."""
    scores = torch.full((3, EXPERTS), bad)
    v, i = rowmax_topk(scores, TOPK)
    for row in i:
        assert sorted(row.tolist()) == list(range(TOPK))
    assert torch.isnan(v).all(), "no real maximum: the row stays poisoned"


@SENTINELS
def test_partial_row_at_or_below_the_sentinel_keeps_k_distinct(bad):
    """Nine finite scores and the rest unusable still names ten experts."""
    torch.manual_seed(11)
    scores = torch.rand(2, EXPERTS)
    scores[:, :EXPERTS - 9] = bad
    v, i = rowmax_topk(scores, TOPK)
    for row in i:
        assert len(set(row.tolist())) == TOPK
    assert not torch.isnan(v[:, 0]).any(), "a finite maximum is not poisoned"


def test_masking_is_below_the_flush_sentinel():
    """The property the two must not share, stated once."""
    assert torch.finfo(torch.float32).min < NEG
    assert torch.finfo(torch.bfloat16).min < NEG


def test_sentinel_is_below_every_real_score_but_above_the_reduce_identity():
    assert NEG < 0.0
    assert NEG > torch.finfo(torch.float32).min


def test_k_equal_to_expert_count():
    torch.manual_seed(3)
    scores = torch.rand(4, 16)
    _assert_matches_reference(scores, k=16)


def test_rejects_k_larger_than_experts():
    with pytest.raises(ValueError):
        rowmax_topk(torch.rand(2, 8), 9)


def test_renormalized_weights_match_the_sort_path():
    """End of the router: renormalized bf16 weights must be bit-identical."""
    torch.manual_seed(11)
    logits = (torch.randn(512, EXPERTS) * 2.0).to(torch.bfloat16)
    scores = logits.float().softmax(dim=-1)

    def renorm(w):
        return (w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-20)).to(
            torch.bfloat16)

    ref_v, _ = torch.topk(scores, TOPK, dim=-1)
    got_v, _ = rowmax_topk(scores, TOPK)
    assert torch.equal(renorm(got_v), renorm(ref_v))
