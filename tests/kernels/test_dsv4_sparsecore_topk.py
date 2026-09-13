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
"""Tests for the Deepseek V4 SparseCore top-k kernel.

`sparsecore_topk` has two execution paths and, before this file, neither had a
test. `_pick_partition` returns P=1 for most shapes and the call takes the
single-stage fast path; only when `b < 32` or `n > 64K` does it split each row
into P slices, run a local top-k per slice, and merge the `P * k_p` candidates
in a second pass. Every GLM-5.2 configuration exercised in practice to date has
run at `--max-model-len=9216` (`n=10240`), where the batch sizes that matter
resolve to P=1 -- so the **two-stage path had never executed at all** until it
was checked deliberately.

It matters because the path is live, not dead code: `streamindex_topk` calls
this from `_common_path`, and `enable_early_exit` defaults to `False`, so there
is no `lax.cond` and no env flag standing between production and this kernel.
Raising the context is all it takes. At `--max-model-len=132096` the call
resolves to P=4 with 33280-word slices and 8192 candidates.

Two properties of the kernel shape these tests, and both rule out the obvious
assertion:

* **Compare selected SCORES as a multiset, never the indices.** The kernel
  returns indices *unsorted* and its own docstring concedes that equal boundary
  scores "may be broken differently". Real score rows carry hundreds of exact
  ties, so asserting on the index set asserts something the kernel never
  promised, and would fail on a correct implementation.
* **Use no relative tolerance.** This is exact selection, not approximation, so
  the bar is `max |score diff| == 0.0`. A relative tolerance would also be
  meaningless against rows whose maximum is exactly 0.0.

`test_comparison_detects_a_wrong_index` is the reason to believe the rest of the
file. A test whose assertion cannot fail reads as evidence while providing none,
so the comparison helper is run a second time against deliberately corrupted
output and is required to reject it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (LANES,
                                                               _align_to,
                                                               _pick_partition,
                                                               sparsecore_topk)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# `n` for a given `--max-model-len`, mirroring `streamindex_topk.py:602-605`:
# `align_to(pages_per_seq, bkv_p) * page_size // 128 * 128` with page_size=1024
# and bkv_p=2 on the GLM path.
N_SHORT = 10240  # --max-model-len 9216
N_128K = 133120  # --max-model-len 132096

SEED = 20260904


def _topk_n(max_model_len: int, page_size: int = 1024, bkv_p: int = 2) -> int:
    pages_per_seq = -(-max_model_len // page_size)
    return _align_to(pages_per_seq, bkv_p) * page_size // 128 * 128


def _reference_scores(scores: np.ndarray, k: int,
                      row_lengths: np.ndarray) -> np.ndarray:
    """Exact top-k scores per row, descending, `-inf` suffix-padded.

    Positions at or beyond `row_lengths`, and `-inf` scores, are never
    eligible.
    """
    out = np.full((scores.shape[0], k), -np.inf, np.float32)
    for i in range(scores.shape[0]):
        row = scores[i, :int(row_lengths[i])]
        eligible = row[row > -np.inf]
        vals = np.sort(eligible)[::-1][:k]
        out[i, :vals.size] = vals
    return out


def _mismatch(scores: np.ndarray, idx: np.ndarray, k: int,
              row_lengths: np.ndarray) -> str | None:
    """`None` if `idx` is a valid exact top-k, else why it is not.

    Returns a reason rather than asserting so that
    `test_comparison_detects_a_wrong_index` can require this to reject bad
    output. An assertion buried in here could not be tested for its ability to
    fire.
    """
    b, n = scores.shape
    ref = _reference_scores(scores, k, row_lengths)
    got = np.full((b, k), -np.inf, np.float32)

    for i in range(b):
        sel = idx[i]
        live = sel[sel >= 0]
        # `-1` padding must be a suffix, not interleaved.
        n_live = live.size
        if not (np.all(idx[i, n_live:] < 0) and np.all(idx[i, :n_live] >= 0)):
            return f"row {i}: -1 padding is not a suffix"
        if n_live and live.max() >= n:
            return f"row {i}: index {live.max()} out of range for n={n}"
        if np.unique(live).size != n_live:
            return f"row {i}: duplicate indices"
        if n_live and live.max() >= 0:
            beyond = live[live >= int(row_lengths[i])]
            if beyond.size:
                return (f"row {i}: index {beyond[0]} at or beyond "
                        f"row_length {int(row_lengths[i])}")
        vals = np.sort(scores[i, live])[::-1] if n_live else np.zeros(0)
        got[i, :vals.size] = vals

    n_ref = np.isfinite(ref).sum(axis=1)
    n_got = np.isfinite(got).sum(axis=1)
    if not (n_ref == n_got).all():
        bad = int(np.flatnonzero(n_ref != n_got)[0])
        return (f"row {bad}: returned {n_got[bad]} valid entries, "
                f"expected {n_ref[bad]}")

    finite = np.isfinite(ref) & np.isfinite(got)
    if finite.any():
        # Exact: this is a selection, so a correct kernel returns the same
        # score values bit for bit. No relative tolerance.
        diff = np.abs(ref[finite] - got[finite]).max()
        if diff != 0.0:
            return f"max score difference {diff!r}, expected exactly 0.0"
    return None


def _scores(kind: str, b: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Score rows, each probing a different way selection can go wrong."""
    if kind == "strict_desc":
        # Tie-free and monotone: the answer is known in closed form, so a
        # kernel that mishandles ordering has nowhere to hide.
        return np.repeat(-np.arange(n, dtype=np.float32)[None, :], b, axis=0)
    if kind == "random":
        return rng.standard_normal((b, n), dtype=np.float32)
    if kind == "heavy_ties":
        # 17 distinct values across the whole row, so the k-th and (k+1)-th
        # scores are equal in every row and every partition boundary lands in a
        # tie. This is the case that makes an index-based assertion wrong.
        # int8 first: `rng.integers` defaults to int64, which is 1 GiB at the
        # 128K shape.
        return rng.integers(0, 17, size=(b, n),
                            dtype=np.int8).astype(np.float32)
    if kind == "all_negative":
        # Every score strictly negative, and that is the whole point.
        # `monotone_key` is the identity for non-negative floats, so the
        # emit-pairs path -- which recovers a score by re-applying the
        # involution at emit time -- is only doing anything at all when a
        # *winner* is negative. Under every other `kind` here the winners are
        # positive and a kernel that dropped the involution entirely would
        # still pass.
        # Continuous and offset off zero, so the row is tie-free and the
        # selection stays non-degenerate.
        return (-np.abs(rng.standard_normal(
            (b, n), dtype=np.float32)) - np.float32(1.0))
    if kind == "all_neg_inf":
        return np.full((b, n), -np.inf, np.float32)
    raise ValueError(kind)


def _run(scores: np.ndarray, k: int, row_lengths: np.ndarray,
         **flags) -> np.ndarray:
    return np.asarray(
        jax.block_until_ready(
            sparsecore_topk(jnp.asarray(scores), k, jnp.asarray(row_lengths),
                            **flags)))


# Shapes covering the single-stage path and both two-stage partition factors
# production can reach. The expected `P` is asserted separately, not assumed --
# see
# `test_partition_factor_is_what_these_tests_assume`.
SHAPES = [
    pytest.param(64, N_SHORT, 2048, 1, id="p1-b64-n10240"),
    pytest.param(16, N_SHORT, 2048, 2, id="p2-b16-n10240"),
    pytest.param(16, N_128K, 2048, 4, id="p4-b16-n133120"),
    pytest.param(64, N_128K, 2048, 4, id="p4-b64-n133120"),
    pytest.param(16, N_128K, 512, 4, id="p4-b16-n133120-k512"),
    # The other end of the production bucket ladder. `_dp_coordinated_step`
    # all-reduce-MAXes the token count across the 16 DP ranks, so a step in
    # which any one rank prefills drags every rank to the top bucket -- 1024 at
    # `--max-num-batched-tokens=1024`. It is as much a production shape as 16
    # is, and it is the one with 64 stage-1 waves rather than 2.
    pytest.param(1024, N_128K, 2048, 4, id="p4-b1024-n133120"),
]


def test_shape_derivation_matches_max_model_len():
    """The 128K shape is derived, not hardcoded, so a page-size change shows."""
    assert _topk_n(9216) == N_SHORT
    assert _topk_n(132096) == N_128K


@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
def test_partition_factor_is_what_these_tests_assume(b, n, k, expected_p):
    """Pin `P` so a change to `_pick_partition` cannot silently retarget these.

    Without this, tuning `MAX_SLICE_WORDS` or the doubling loop could collapse
    every case below to the single-stage path, and the suite would keep passing
    while testing nothing about the two-stage merge.
    """
    assert _pick_partition(b, n, k) == expected_p


@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
@pytest.mark.parametrize("kind", ["strict_desc", "random", "heavy_ties"])
def test_exact_topk_full_rows(b, n, k, expected_p, kind):
    rng = np.random.default_rng(SEED)
    scores = _scores(kind, b, n, rng)
    row_lengths = np.full(b, n, np.int32)
    assert _mismatch(scores, _run(scores, k, row_lengths), k,
                     row_lengths) is None


@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
def test_exact_topk_ragged_rows_straddle_partition_boundaries(
        b, n, k, expected_p):
    """Ragged lengths are the chunked-prefill case and the two-stage risk.

    `_effective_row_lengths` gives every row in a chunk a different amount of
    visible context, so partitions are routinely partly or wholly empty. Edges
    exactly at `n_p` are where a stage-1 slice contributes nothing and stage 2
    must cope with all-padding candidate blocks.
    """
    n_p = n // expected_p
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, n, rng)
    edges = sorted({
        min(max(e, 0), n)
        for e in (0, 1, LANES, k, n_p - 1, n_p, n_p + 1, 2 * n_p, 2 * n_p + 17,
                  3 * n_p, n - 1, n)
    })
    row_lengths = np.array([edges[i % len(edges)] for i in range(b)], np.int32)
    assert _mismatch(scores, _run(scores, k, row_lengths), k,
                     row_lengths) is None


@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
def test_all_neg_inf_rows_return_no_selection(b, n, k, expected_p):
    """`-inf` is never eligible, so every output slot must be `-1`."""
    rng = np.random.default_rng(SEED)
    scores = _scores("all_neg_inf", b, n, rng)
    row_lengths = np.full(b, n, np.int32)
    idx = _run(scores, k, row_lengths)
    assert (idx == -1).all(), f"expected all -1, got {np.unique(idx)[:8]}"
    assert _mismatch(scores, idx, k, row_lengths) is None


def test_topk_concentrated_in_the_last_partition():
    """The two-stage path's own failure mode.

    All the mass sits in the final slice, so three of the four stage-1
    candidate blocks are pure padding and stage 2 has to discard them without
    letting a padded entry outrank a real one.
    """
    b, n, k, p = 16, N_128K, 2048, 4
    assert _pick_partition(b, n, k) == p
    rng = np.random.default_rng(SEED)
    scores = np.full((b, n), -1e30, np.float32)
    scores[:, -(k + 64):] = rng.standard_normal((b, k + 64)).astype(np.float32)
    row_lengths = np.full(b, n, np.int32)
    assert _mismatch(scores, _run(scores, k, row_lengths), k,
                     row_lengths) is None


TWO_STAGE_SHAPES = [p for p in SHAPES if p.values[3] > 1]


@pytest.mark.parametrize("b,n,k,expected_p", TWO_STAGE_SHAPES)
@pytest.mark.parametrize("kind", ["random", "heavy_ties", "all_negative"])
def test_emit_pairs_is_bit_identical_to_the_incumbent(b, n, k, expected_p,
                                                      kind):
    """The shipped path may not change the answer, not even a tie-break.

    This is a stronger assertion than re-running `_mismatch` on the shipped
    path, and a much cheaper one -- the reference sort dominates that helper's
    cost. It is also the assertion that matches what the change claims to be: a
    reformulation that moves a value out of an XLA gather and into the kernel
    that already had it, not a different selection rule.

    Note the polarity: `_reference_stage2_gather=True` is now the *baseline*
    and the default is what ships. This test is the reason that seam still
    exists -- delete it and nothing establishes bit-identity any more, because
    `_mismatch` compares score multisets and so cannot see a tie-break change.

    `heavy_ties` is here because that is where a reformulation is most likely
    to differ without being *wrong* -- and it still must not, because the two
    paths visit candidates in the same order.

    `all_negative` is here because without it this test cannot fail.
    `monotone_key` is the identity on non-negative floats, so under `random`
    and `heavy_ties` -- whose winners are the top 2048 of a distribution
    centred at or above zero -- a kernel that dropped the involution altogether
    stays bit-identical. Measured: mutating `monotone_key` to the identity left
    the whole file green.
    """
    assert _pick_partition(b, n, k) == expected_p > 1, (
        "this test is meaningless on the single-stage path, where the two "
        "formulations are the same code")
    rng = np.random.default_rng(SEED)
    scores = _scores(kind, b, n, rng)
    row_lengths = np.full(b, n, np.int32)

    base = _run(scores, k, row_lengths, _reference_stage2_gather=True)
    if kind == "all_negative":
        # Non-vacuity, asserted rather than assumed: the case earns its place
        # only if the winners really are negative *and* the selection is a
        # choice. `row_length <= k` would make every live entry a winner
        # regardless of its score, which is how this file's ragged rows manage
        # to contain 2151 negative winners and still discriminate nothing.
        assert k < int(row_lengths.min()), (
            "selection is degenerate: every live entry wins, so score values "
            "never enter the comparison")
        won = np.take_along_axis(scores, base, axis=1)
        assert (won < 0).all(), (
            f"{int((won >= 0).sum())} of {won.size} winners are non-negative, "
            f"so `monotone_key` is the identity for them and this case does "
            f"not exercise the involution it exists to exercise")
    got = _run(scores, k, row_lengths)
    n_diff = int((base != got).sum())
    assert n_diff == 0, (
        f"the shipped path differs from the incumbent in {n_diff} of "
        f"{base.size} entries; first at {np.argwhere(base != got)[0]}")


@pytest.mark.parametrize("b,n,k,expected_p", TWO_STAGE_SHAPES)
def test_emit_pairs_survives_ragged_rows(b, n, k, expected_p):
    """Ragged lengths, against the reference *and* against the incumbent.

    The ragged case is where the padding convention actually differs between
    the two formulations: the shipped path has to write `-inf` into the pad
    slots that `fill_body` fills with `-1`, or a padded candidate can outrank a
    real one in stage 2. So it gets both assertions, and neither is redundant.
    Bit-identity alone would still hold if both paths were wrong in the same
    way; `_mismatch` alone compares score multisets and cannot see a tie-break
    difference. Full-row bit-identity is the test above -- this is the only
    place the two formulations are compared under ragged rows.
    """
    n_p = n // expected_p
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, n, rng)
    edges = sorted({
        min(max(e, 0), n)
        for e in (0, 1, LANES, k, n_p - 1, n_p, n_p + 1, 2 * n_p, 2 * n_p + 17,
                  3 * n_p, n - 1, n)
    })
    row_lengths = np.array([edges[i % len(edges)] for i in range(b)], np.int32)
    idx = _run(scores, k, row_lengths)
    assert _mismatch(scores, idx, k, row_lengths) is None

    base = _run(scores, k, row_lengths, _reference_stage2_gather=True)
    n_diff = int((base != idx).sum())
    assert n_diff == 0, (
        f"the shipped path differs from the incumbent on ragged rows in "
        f"{n_diff} of {base.size} entries; first at "
        f"{np.argwhere(base != idx)[0]}")


@pytest.mark.parametrize("corruption",
                         ["worst_index", "drop_one", "duplicate", "negative"])
def test_comparison_detects_a_wrong_index(corruption):
    """The meta-test: prove `_mismatch` can reject bad output.

    Every other assertion in this file is `_mismatch(...) is None`, which is
    worth exactly as much as this test. Four independent corruptions, each of
    which a correct kernel would never produce.

    `worst_index` swaps in the row's *minimum* eligible column rather than
    shifting an index by one. Under `heavy_ties` an adjacent column very often
    carries an identical score, so a one-off shift leaves the score multiset
    unchanged -- the comparison would rightly accept it, and the control would
    look broken when it was the perturbation that was too weak.
    """
    b, n, k = 16, N_128K, 2048
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, n, rng)
    row_lengths = np.full(b, n, np.int32)
    idx = _run(scores, k, row_lengths)
    assert _mismatch(scores, idx, k, row_lengths) is None, "baseline must pass"

    bad = idx.copy()
    if corruption == "worst_index":
        bad[0, 0] = int(np.argmin(scores[0, :int(row_lengths[0])]))
    elif corruption == "drop_one":
        bad[0, 0] = -1  # breaks the suffix-padding invariant and the count
    elif corruption == "duplicate":
        bad[0, 0] = int(bad[0, 1])
    elif corruption == "negative":
        bad[0, 0] = n + 5  # out of range

    assert _mismatch(scores, bad, k, row_lengths) is not None, (
        f"corruption {corruption!r} was NOT detected -- the comparison used by "
        f"every other test in this file cannot fail, so those tests prove "
        f"nothing")
