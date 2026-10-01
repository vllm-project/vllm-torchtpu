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
single-stage fast path; only when `b < 32` or `n > 32K` does it split each row
into P slices, run a local top-k per slice, and merge the `P * k_p` candidates
in a second pass. Every GLM-5.2 configuration exercised in practice to date has
run at `--max-model-len=9216` (`n=10240`), where the batch sizes that matter
resolve to P=1 -- so the **two-stage path had never executed at all** until it
was checked deliberately.

It matters because the path is live, not dead code: `streamindex_topk` calls
this from `_common_path`, and `enable_early_exit` defaults to `False`, so there
is no `lax.cond` and no env flag standing between production and this kernel.
Raising the context is all it takes. At `--max-model-len=132096` the call
resolves to P=8 with 16640-word slices and 16384 candidates.

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

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_module
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    LANES,
    MAX_SLICE_WORDS,
    _align_to,
    _pick_partition,
    sparsecore_topk,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# `n` for a given `--max-model-len`, mirroring `streamindex_topk.py:602-605`:
# `align_to(pages_per_seq, bkv_p) * page_size // 128 * 128` with page_size=1024
# and bkv_p=2 on the GLM path.
N_SHORT = 10240  # --max-model-len 9216
N_128K = 133120  # --max-model-len 132096
N_1M = 1048576  # --max-model-len 1048576, the GLM-5.2 DCP=16 shape
N_256K = 264192  # --max-model-len 263168, the GLM-5.2 256K-in / 1K-out shape

SEED = 20260904


def _topk_n(max_model_len: int, page_size: int = 1024, bkv_p: int = 2) -> int:
    pages_per_seq = -(-max_model_len // page_size)
    return _align_to(pages_per_seq, bkv_p) * page_size // 128 * 128


def _reference_scores(
    scores: np.ndarray, k: int, row_lengths: np.ndarray
) -> np.ndarray:
    """Exact top-k scores per row, descending, `-inf` suffix-padded.

    Positions at or beyond `row_lengths`, and `-inf` scores, are never
    eligible.
    """
    out = np.full((scores.shape[0], k), -np.inf, np.float32)
    for i in range(scores.shape[0]):
        row = scores[i, : int(row_lengths[i])]
        eligible = row[row > -np.inf]
        vals = np.sort(eligible)[::-1][:k]
        out[i, : vals.size] = vals
    return out


def _mismatch(
    scores: np.ndarray, idx: np.ndarray, k: int, row_lengths: np.ndarray
) -> str | None:
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
                return (
                    f"row {i}: index {beyond[0]} at or beyond "
                    f"row_length {int(row_lengths[i])}"
                )
        vals = np.sort(scores[i, live])[::-1] if n_live else np.zeros(0)
        got[i, : vals.size] = vals

    n_ref = np.isfinite(ref).sum(axis=1)
    n_got = np.isfinite(got).sum(axis=1)
    if not (n_ref == n_got).all():
        bad = int(np.flatnonzero(n_ref != n_got)[0])
        return f"row {bad}: returned {n_got[bad]} valid entries, expected {n_ref[bad]}"

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
        # This is also the only `kind` whose winners are negative, so it is
        # what covers `monotone_key` -- the involution is the identity on
        # non-negative floats, and `random`/`heavy_ties` are centred at or
        # above zero. Measured: dropping the involution at stage 2's emit site
        # fails this `kind` on all five two-stage shapes (and nothing on the
        # single-stage one, which has no emit site). The only other test that
        # catches it is `test_topk_concentrated_in_the_last_partition`, via its
        # negative `-1e30` padding fill.
        return np.repeat(-np.arange(n, dtype=np.float32)[None, :], b, axis=0)
    if kind == "random":
        return rng.standard_normal((b, n), dtype=np.float32)
    if kind == "heavy_ties":
        # 17 distinct values across the whole row, so the k-th and (k+1)-th
        # scores are equal in every row and every partition boundary lands in a
        # tie. This is the case that makes an index-based assertion wrong.
        # int8 first: `rng.integers` defaults to int64, which is 1 GiB at the
        # 128K shape.
        return rng.integers(0, 17, size=(b, n), dtype=np.int8).astype(np.float32)
    if kind == "all_neg_inf":
        return np.full((b, n), -np.inf, np.float32)
    raise ValueError(kind)


def _run(scores: np.ndarray, k: int, row_lengths: np.ndarray) -> np.ndarray:
    return np.asarray(
        jax.block_until_ready(
            sparsecore_topk(jnp.asarray(scores), k, jnp.asarray(row_lengths))
        )
    )


# Shapes covering the single-stage path and both two-stage partition factors
# production can reach. The expected `P` is asserted separately, not assumed --
# see
# `test_partition_factor_is_what_these_tests_assume`. `P` at the 128K shape is
# 8 rather than 4 because `MAX_SLICE_WORDS` is the *single-buffer* budget and
# the resident slice is now double-buffered by `emit_pipeline`, so `p_min`
# doubled.
SHAPES = [
    pytest.param(64, N_SHORT, 2048, 1, id="p1-b64-n10240"),
    pytest.param(16, N_SHORT, 2048, 2, id="p2-b16-n10240"),
    pytest.param(16, N_128K, 2048, 8, id="p8-b16-n133120"),
    pytest.param(64, N_128K, 2048, 8, id="p8-b64-n133120"),
    pytest.param(16, N_128K, 512, 8, id="p8-b16-n133120-k512"),
    # The other end of the production bucket ladder. `_dp_coordinated_step`
    # all-reduce-MAXes the token count across the 16 DP ranks, so a step in
    # which any one rank prefills drags every rank to the top bucket -- 1024 at
    # `--max-num-batched-tokens=1024`. It is as much a production shape as 16
    # is, and it is the one that needs 64x as many stage-1 waves as `b=16`.
    pytest.param(1024, N_128K, 2048, 8, id="p8-b1024-n133120"),
]


def test_shape_derivation_matches_max_model_len():
    """The 128K shape is derived, not hardcoded, so a page-size change shows."""
    assert _topk_n(9216) == N_SHORT
    assert _topk_n(132096) == N_128K
    assert _topk_n(1048576, page_size=256) == N_1M
    assert _topk_n(263168) == N_256K


@pytest.mark.parametrize("b,k", [(64, 2048), (128, 2048)])
def test_exact_topk_when_stage_two_needs_its_own_partition(b, k):
    """`p * k_p` past one slice must recurse, not overflow SparseCore.

    At 1M context `p` reaches 32, so stage 2 sees 65536 candidates. Feeding
    that straight to `_sc_topk_direct` fails to compile with E3000
    CompileTimeSparseCoreAllocationFailure in tile_spmem. Below `b=32` the
    single stage-1 wave keeps the same width inside budget, so the batch sizes
    here are the ones that actually reproduce it.
    """
    p = _pick_partition(b, N_1M, k)
    assert p * min(k, N_1M // p) > MAX_SLICE_WORDS, (
        f"{b=} {k=} no longer reaches the recursive stage-2 path (p={p})"
    )
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, N_1M, rng)
    row_lengths = np.full(b, N_1M, np.int32)
    assert _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None


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
    assert _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None


@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
def test_exact_topk_ragged_rows_straddle_partition_boundaries(b, n, k, expected_p):
    """Ragged lengths are the chunked-prefill case and the two-stage risk.

    `_effective_row_lengths` gives every row in a chunk a different amount of
    visible context, so partitions are routinely partly or wholly empty. Edges
    exactly at `n_p` are where a stage-1 slice contributes nothing and stage 2
    must cope with all-padding candidate blocks.
    """
    n_p = n // expected_p
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, n, rng)
    edges = sorted(
        {
            min(max(e, 0), n)
            for e in (
                0,
                1,
                LANES,
                k,
                n_p - 1,
                n_p,
                n_p + 1,
                2 * n_p,
                2 * n_p + 17,
                3 * n_p,
                n - 1,
                n,
            )
        }
    )
    row_lengths = np.array([edges[i % len(edges)] for i in range(b)], np.int32)
    assert _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None


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

    All the mass sits in the final slice, so all but one of the stage-1
    candidate blocks are pure padding and stage 2 has to discard them without
    letting a padded entry outrank a real one.
    """
    b, n, k, p = 16, N_128K, 2048, 8
    assert _pick_partition(b, n, k) == p
    rng = np.random.default_rng(SEED)
    scores = np.full((b, n), -1e30, np.float32)
    scores[:, -(k + 64) :] = rng.standard_normal((b, k + 64)).astype(np.float32)
    row_lengths = np.full(b, n, np.int32)
    assert _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None


@pytest.mark.parametrize(
    "corruption", ["worst_index", "drop_one", "duplicate", "negative"]
)
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
        bad[0, 0] = int(np.argmin(scores[0, : int(row_lengths[0])]))
    elif corruption == "drop_one":
        bad[0, 0] = -1  # breaks the suffix-padding invariant and the count
    elif corruption == "duplicate":
        bad[0, 0] = int(bad[0, 1])
    elif corruption == "negative":
        bad[0, 0] = n + 5  # out of range

    assert _mismatch(scores, bad, k, row_lengths) is not None, (
        f"corruption {corruption!r} was NOT detected -- the comparison used by "
        f"every other test in this file cannot fail, so those tests prove "
        f"nothing"
    )


@pytest.mark.parametrize(
    "b,n,k",
    [
        (32, 2048, 256),  # single_stage
        (4, 8192, 256),  # two_stage
    ],
)
def test_return_scores(b: int, n: int, k: int):
    """Verifies that return_scores=True returns matching scores for winners and
    -inf for pad.
    """
    np.random.seed(800 + b + n + k)
    scores = np.random.randn(b, n).astype(np.float32)
    row_lengths = np.random.randint(k, n + 1, size=(b,), dtype=np.int32)

    actual_idxs, actual_scores_bits = sparsecore_topk(
        jnp.array(scores),
        k,
        row_lengths=jnp.array(row_lengths),
        return_scores=True,
    )
    assert _mismatch(scores, actual_idxs, k, row_lengths) is None

    actual_scores = np.asarray(
        jax.lax.bitcast_convert_type(actual_scores_bits, jnp.float32)
    )
    for r in range(b):
        for i in range(k):
            idx = int(actual_idxs[r, i])
            score_val = actual_scores[r, i]
            if idx >= 0:
                np.testing.assert_allclose(score_val, scores[r, idx], rtol=1e-5)
            else:
                assert score_val == -np.inf


# --------------------------------------------------------------------------- #
# Scheduling groups.
# --------------------------------------------------------------------------- #
# `scheduling_group_id` puts stage 1 in an XLA scheduling group and
# `stage2_scheduling_group_id` puts stage 2 in a *different* one, so a caller
# pipelining several chunks can overlap each stage with a different chunk's
# TensorCore work. Stage 2 only exists on the two-stage path.
#
# Both ids are scheduling hints, so there are exactly two ways they can be
# wrong: the annotation lands on the wrong stage (or leaks onto a stage the
# caller left unannotated, which would pin two stages into one group and
# serialise the overlap it was added to create), or it perturbs the selection.
# One test each. Which stage gets which id is decided before any kernel is
# built, so that half is a `cpu_test` and only the other two need a TPU.

# `b=16, n=N_128K` is the two-stage shape; `b=64, n=N_SHORT` is single-stage.
TWO_STAGE = (16, N_128K, 2048)
SINGLE_STAGE = (64, N_SHORT, 2048)


def _stage_group_ids(monkeypatch, b, n, k, **kwargs) -> list[int | None]:
    """The `scheduling_group_id` each kernel launch receives, in order.

    Every launch -- the single-stage path, stage 1 and stage 2 -- goes through
    `_sc_topk_rows`, so stubbing it records one entry per stage.

    Calls `sparsecore_topk.__wrapped__` rather than the jitted wrapper so the
    stub is guaranteed to be the function that runs: a cached trace from an
    earlier test would otherwise re-run the real kernel and record nothing.
    """
    seen: list[int | None] = []

    def _stub(
        words_flat,
        k_,
        row_lens,
        row_offs,
        *,
        slice_len,
        scheduling_group_id,
        name,
        row_bases=None,
        payload_flat=None,
        flat_out=False,
    ):
        seen.append(scheduling_group_id)
        out = jnp.zeros((row_lens.shape[0], k_), jnp.int32)
        if flat_out:
            out = out.reshape(-1)
        return out, out

    monkeypatch.setattr(sc_module, "_sc_topk_rows", _stub)
    sc_module.sparsecore_topk.__wrapped__(jnp.zeros((b, n), jnp.float32), k, **kwargs)
    return seen


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "shape,kwargs,expected",
    [
        # Each stage gets its own group, and stage 2 gets the stage-2 one.
        (TWO_STAGE, dict(scheduling_group_id=3, stage2_scheduling_group_id=4), [3, 4]),
        # Stage 2 unannotated: it must NOT inherit stage 1's group.
        (TWO_STAGE, dict(scheduling_group_id=3), [3, None]),
        # Stage 1 unannotated, stage 2 annotated: the two are independent.
        (TWO_STAGE, dict(stage2_scheduling_group_id=4), [None, 4]),
        (TWO_STAGE, {}, [None, None]),
        # One stage exists, so the stage-2 id is accepted and unused.
        (SINGLE_STAGE, dict(scheduling_group_id=3, stage2_scheduling_group_id=4), [3]),
        (SINGLE_STAGE, dict(stage2_scheduling_group_id=4), [None]),
    ],
    ids=["both", "stage1_only", "stage2_only", "neither", "p1_both", "p1_stage2_only"],
)
def test_each_stage_gets_the_group_id_it_was_given(
    monkeypatch, shape, kwargs, expected
):
    b, n, k = shape
    assert _pick_partition(b, n, k) == (8 if len(expected) == 2 else 1)
    assert _stage_group_ids(monkeypatch, b, n, k, **kwargs) == expected


def test_scheduling_group_ids_reach_the_lowered_hlo():
    """The stub test above proves the plumbing; this proves the plumbing does
    something. Both ids have to survive as frontend attributes, or the stages
    are annotated only in the source."""
    b, n, k = TWO_STAGE
    assert _pick_partition(b, n, k) == 8
    scores = jnp.zeros((b, n), jnp.float32)
    row_lengths = jnp.full((b,), n, jnp.int32)

    def _text(**kwargs):
        return sparsecore_topk.lower(scores, k, row_lengths, **kwargs).as_text()

    both = _text(scheduling_group_id=3, stage2_scheduling_group_id=4)
    assert '_scheduling_group_id = "3"' in both
    assert '_scheduling_group_id = "4"' in both

    # Nothing is annotated unless it was asked for.
    stage1_only = _text(scheduling_group_id=3)
    assert '_scheduling_group_id = "3"' in stage1_only
    assert '_scheduling_group_id = "4"' not in stage1_only
    assert "_scheduling_group_id" not in _text()


@pytest.mark.parametrize("kind", ["random", "heavy_ties"])
def test_scheduling_group_ids_do_not_change_the_selection(kind):
    """A scheduling hint that changed the answer would be a correctness bug."""
    b, n, k = TWO_STAGE
    rng = np.random.default_rng(SEED)
    scores = _scores(kind, b, n, rng)
    row_lengths = np.full(b, n, np.int32)

    plain = _run(scores, k, row_lengths)
    grouped = np.asarray(
        jax.block_until_ready(
            sparsecore_topk(
                jnp.asarray(scores),
                k,
                jnp.asarray(row_lengths),
                scheduling_group_id=3,
                stage2_scheduling_group_id=4,
            )
        )
    )

    assert _mismatch(scores, grouped, k, row_lengths) is None
    # Indices may legitimately differ on ties, the selected scores may not.
    np.testing.assert_array_equal(
        np.sort(np.take_along_axis(scores, np.maximum(plain, 0), axis=1), axis=1),
        np.sort(np.take_along_axis(scores, np.maximum(grouped, 0), axis=1), axis=1),
    )


# --------------------------------------------------------------------------- #
# The 256K deployment shape: length-bounded, rebalanced stage 1.
# --------------------------------------------------------------------------- #
# Stage 1 cuts each row into P sub-rows of a traced width
# `w = align16(cdiv(max(row_lengths), P))` and DMAs only `align16(sub_len)`
# words of each, so the batch's longest row -- not `n` -- sets how much is
# read, and empty rows (decode's padding rows) read nothing. The risks are
# therefore at row lengths that land on sub-row and 16-lane edges, at
# batches whose longest row is short (small `w`), and at empty rows. `b=16`
# is the decode bucket and `b=1024` the top prefill bucket.

DEPLOY_K = 2048
# 16512 = N_256K / 16 is the static sub-row width at P = 16.
DEPLOY_EDGES = (0, 1, 2047, 2048, 16512, 16513, 131072, N_256K)


def _deploy_scores(b: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random rows with the three tie / `-inf` patterns interleaved."""
    x = rng.standard_normal((b, n), dtype=np.float32)
    x[1::4] = np.round(x[1::4] * 2) / 2  # heavy ties: splits at the k-th score
    x[2::4, ::7] = -np.inf  # `-inf` entries are never eligible
    x[3::8] = 0.5  # all-equal rows
    return x


def _run_with_scores(scores, k, row_lengths):
    idx, bits = jax.block_until_ready(
        sparsecore_topk(
            jnp.asarray(scores), k, jnp.asarray(row_lengths), return_scores=True
        )
    )
    return np.asarray(idx), np.asarray(bits)


def _score_bits_mismatch(scores, idx, bits) -> str | None:
    """`return_scores` must pair each winner with its score's bits, and pad
    slots with the bits of `-inf`."""
    neg_inf_bits = np.float32(-np.inf).view(np.int32)
    rows = np.arange(scores.shape[0])[:, None]
    want = np.where(
        idx >= 0, scores[rows, np.maximum(idx, 0)].view(np.int32), neg_inf_bits
    )
    bad = np.argwhere(bits != want)
    if bad.size:
        return f"row {bad[0][0]} slot {bad[0][1]}: score bits do not match"
    return None


@pytest.mark.parametrize("b", [16, 1024])
def test_partition_factor_at_the_deployment_shape(b):
    assert _pick_partition(b, N_256K, DEPLOY_K) == 16


@pytest.mark.parametrize("b", [16, 1024])
def test_exact_topk_deployment_edge_lengths(b):
    """Every edge length, twice, then random lengths; longest row is `n`."""
    rng = np.random.default_rng(SEED)
    scores = _deploy_scores(b, N_256K, rng)
    row_lengths = rng.integers(0, N_256K + 1, size=b).astype(np.int32)
    edges = list(DEPLOY_EDGES) * 2
    row_lengths[: len(edges)] = edges
    idx, bits = _run_with_scores(scores, DEPLOY_K, row_lengths)
    assert _mismatch(scores, idx, DEPLOY_K, row_lengths) is None
    assert _score_bits_mismatch(scores, idx, bits) is None


@pytest.mark.parametrize("max_len", [2047, 2048, 16513, 131072])
def test_exact_topk_deployment_short_batch(max_len):
    """A batch whose longest row is short shrinks `w` below the static width.

    `b=16` (decode bucket, one stage-1 wave) with the edges clipped to
    `max_len`: sub-row boundaries move to multiples of the traced `w`.
    """
    b = 16
    rng = np.random.default_rng(SEED + max_len)
    scores = _deploy_scores(b, N_256K, rng)
    row_lengths = np.array(
        [min(DEPLOY_EDGES[i % len(DEPLOY_EDGES)], max_len) for i in range(b)],
        np.int32,
    )
    row_lengths[-1] = max_len
    idx, bits = _run_with_scores(scores, DEPLOY_K, row_lengths)
    assert _mismatch(scores, idx, DEPLOY_K, row_lengths) is None
    assert _score_bits_mismatch(scores, idx, bits) is None


def test_exact_topk_deployment_prefill_short_rows():
    """The top prefill bucket at modest context: every row under 34817."""
    b = 1024
    rng = np.random.default_rng(SEED)
    scores = _deploy_scores(b, N_256K, rng)
    row_lengths = rng.integers(0, 34817, size=b).astype(np.int32)
    row_lengths[:4] = [0, 1, 2048, 2049]
    idx, bits = _run_with_scores(scores, DEPLOY_K, row_lengths)
    assert _mismatch(scores, idx, DEPLOY_K, row_lengths) is None
    assert _score_bits_mismatch(scores, idx, bits) is None


@pytest.mark.parametrize("real_len", [1, 2048, 131072, N_256K])
def test_exact_topk_decode_bucket_one_real_row(real_len):
    """Decode: one real row of 16, the 15 padding rows empty (no DMA)."""
    b = 16
    rng = np.random.default_rng(SEED)
    scores = _deploy_scores(b, N_256K, rng)
    row_lengths = np.zeros(b, np.int32)
    row_lengths[0] = real_len
    idx, bits = _run_with_scores(scores, DEPLOY_K, row_lengths)
    assert _mismatch(scores, idx, DEPLOY_K, row_lengths) is None
    assert _score_bits_mismatch(scores, idx, bits) is None
    assert (idx[1:] == -1).all(), "empty rows must select nothing"


@pytest.mark.parametrize("b", [16, 1024])
def test_all_rows_empty_select_nothing(b):
    """Every row length 0: no DMA anywhere, every slot `-1`."""
    rng = np.random.default_rng(SEED)
    scores = _deploy_scores(b, N_256K, rng)
    row_lengths = np.zeros(b, np.int32)
    idx = _run(scores, DEPLOY_K, row_lengths)
    assert (idx == -1).all(), f"expected all -1, got {np.unique(idx)[:8]}"


def test_exact_topk_single_stage_edge_lengths():
    """The single-stage path (P == 1) reads `align16(len)` words per row."""
    b, n = 64, 32768
    assert _pick_partition(b, n, DEPLOY_K) == 1
    rng = np.random.default_rng(SEED)
    scores = _deploy_scores(b, n, rng)
    row_lengths = rng.integers(0, n + 1, size=b).astype(np.int32)
    edges = [min(e, n) for e in (0, 1, 15, 17, 2047, 2048, n - 1, n)]
    row_lengths[: len(edges)] = edges
    idx, bits = _run_with_scores(scores, DEPLOY_K, row_lengths)
    assert _mismatch(scores, idx, DEPLOY_K, row_lengths) is None
    assert _score_bits_mismatch(scores, idx, bits) is None


def test_all_rows_empty_then_decode_in_the_same_process():
    """The idle-rank decode bucket (b=16, every row empty) issues no DMA; a
    normal decode right after must still launch and be exact."""
    b = 16
    rng = np.random.default_rng(SEED + 1)
    scores = _deploy_scores(b, N_256K, rng)
    empty = _run(scores, DEPLOY_K, np.zeros(b, np.int32))
    assert (empty == -1).all(), f"expected all -1, got {np.unique(empty)[:8]}"
    row_lengths = np.zeros(b, np.int32)
    row_lengths[0] = 131072
    for _ in range(2):
        idx, bits = _run_with_scores(scores, DEPLOY_K, row_lengths)
        assert _mismatch(scores, idx, DEPLOY_K, row_lengths) is None
        assert _score_bits_mismatch(scores, idx, bits) is None


# --------------------------------------------------------------------------- #
# Two-stage path with in-kernel index carry.
# --------------------------------------------------------------------------- #
# When the `P * k_p` candidates fit one stage-2 slice, stage 1 emits global
# column indices (sub-row base `j*w` added in-kernel) and stage 2 compacts
# those indices as a payload alongside the keys; no XLA remap follows. The
# risks are a payload/key misalignment (caught by the score-bits pairing) and
# a wrong base (caught by the multiset check). `n = 270336` is the next
# `--max-model-len` bucket above 256K.
#
# Not covered: `b = 1, n = N_256K` (P = 32, `P * k_p > MAX_SLICE_WORDS`) takes
# the recursive fallback, which recurses without terminating in the code this
# path was added to. It is unreachable in serving (the minimum token bucket is
# 16) and the fallback is unchanged here.

N_270K = 270336
CARRY_SHAPES = [
    pytest.param(b, n, id=f"b{b}-n{n}")
    for n in (N_256K, N_270K)
    for b in (16, 128, 1024)
]


def _carry_case(kind: str, b: int, n: int, rng: np.random.Generator):
    scores = _deploy_scores(b, n, rng)
    row_lengths = rng.integers(0, n + 1, size=b).astype(np.int32)
    if kind == "edges":
        edges = [min(e, n) for e in DEPLOY_EDGES] + [n]
        edges = (edges * 2)[:b]
        row_lengths[: len(edges)] = edges
    elif kind == "decode":  # one real row, the rest decode padding rows
        row_lengths[:] = 0
        row_lengths[0] = min(262144, n)
    elif kind == "short":  # prefill-like mid lengths + ties
        row_lengths = rng.integers(0, 34817, size=b).astype(np.int32)
        row_lengths[:4] = [0, 1, 2048, 2049]
        scores[::5] = np.round(scores[::5] * 4) / 4
    elif kind == "full":
        row_lengths[:] = n
        scores[::3] = np.round(scores[::3] * 8) / 8
    return scores, row_lengths


@pytest.mark.parametrize("b,n", CARRY_SHAPES)
def test_index_carry_path_is_taken(b, n):
    p = _pick_partition(b, n, DEPLOY_K)
    k_p = min(DEPLOY_K, n // p)
    assert p > 1
    assert p * k_p <= MAX_SLICE_WORDS and (p * k_p) % LANES == 0


@pytest.mark.parametrize("b,n", CARRY_SHAPES)
@pytest.mark.parametrize("kind", ["edges", "decode", "short", "random", "full"])
def test_exact_topk_index_carry(b, n, kind):
    """Exact selection, winner/score pairing, and `return_scores=False`
    returning the same indices, on the index-carry two-stage path."""
    rng = np.random.default_rng(SEED + b + n)
    scores, row_lengths = _carry_case(kind, b, n, rng)
    idx, bits = _run_with_scores(scores, DEPLOY_K, row_lengths)
    assert _mismatch(scores, idx, DEPLOY_K, row_lengths) is None
    assert _score_bits_mismatch(scores, idx, bits) is None
    np.testing.assert_array_equal(_run(scores, DEPLOY_K, row_lengths), idx)
    if kind == "decode":
        assert (idx[1:] == -1).all(), "empty rows must select nothing"
