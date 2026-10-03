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

`sparsecore_topk` executes in a single kernel launch for partition factors
`P in {1, 2, 4, 8, 16}` (`P == 1` per-subcore or `P > 1` cooperative
`VMEM_SHARED` global radix histogram), and falls back to the zero-gather
two-stage merge (`_sc_topk_two_stage`, carrying stage-1 indices as the stage-2
`values` payload) when `P > 16` (e.g. `N_1M = 1048576` where `P = 32`).

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

import ast
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_module
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    LANES,
    MAX_SLICE_WORDS,
    _align_to,
    _pick_partition,
    _select_p,
    sparsecore_topk,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# `n` for a given `--max-model-len`, mirroring `streamindex_topk.py:602-605`:
# `align_to(pages_per_seq, bkv_p) * page_size // 128 * 128` with page_size=1024
# and bkv_p=2 on the GLM path.
N_SHORT = 10240  # --max-model-len 9216
N_128K = 133120  # --max-model-len 132096
N_1M = 1048576  # --max-model-len 1048576, the GLM-5.2 DCP=16 shape

SEED = 20260904


def _require_sparsecore():
    """Skip hardware execution tests when no SparseCore TPU device is present."""
    try:
        if pltpu.get_tpu_info().sparse_core is not None:
            return
    except Exception:
        pass
    pytest.skip("needs SparseCore")


def _tpu_lowering_context():
    """Provide a TPU7x AbstractMesh on CPU hosts so `.lower()` can target TPU."""
    if jax.devices()[0].platform == "tpu":
        return nullcontext()
    ad = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=2, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=ad)
    return jax.sharding.use_abstract_mesh(mesh)


def _lower_to_tpu_text(fn, scores, k: int, *args, **kwargs) -> str:
    with _tpu_lowering_context():
        traced = jax.jit(lambda s, *a: fn(s, k, *a, **kwargs)).trace(scores, *args)
        return traced.lower(lowering_platforms=("tpu",)).as_text()


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
        return np.repeat(-np.arange(n, dtype=np.float32)[None, :], b, axis=0)
    if kind == "random":
        return rng.standard_normal((b, n), dtype=np.float32)
    if kind == "heavy_ties":
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


# Shapes covering `P == 1` and cooperative `VMEM_SHARED` `P in {2, 8}`
# partition factors production can reach.
SHAPES = [
    pytest.param(64, N_SHORT, 2048, 1, id="p1-b64-n10240"),
    pytest.param(16, N_SHORT, 2048, 2, id="p2-b16-n10240"),
    pytest.param(16, N_128K, 2048, 8, id="p8-b16-n133120"),
    pytest.param(64, N_128K, 2048, 8, id="p8-b64-n133120"),
    pytest.param(16, N_128K, 512, 8, id="p8-b16-n133120-k512"),
    pytest.param(1024, N_128K, 2048, 8, id="p8-b1024-n133120"),
]


@pytest.mark.cpu_test
def test_shape_derivation_matches_max_model_len():
    """The 128K shape is derived, not hardcoded, so a page-size change shows."""
    assert _topk_n(9216) == N_SHORT
    assert _topk_n(132096) == N_128K
    assert _topk_n(1048576, page_size=256) == N_1M


@pytest.mark.parametrize("b,k", [(64, 2048), (128, 2048)])
def test_exact_topk_when_stage_two_needs_its_own_partition(b, k):
    """`p > 16` at 1M context uses the two-stage fallback and stage 2 runs VMEM_SHARED."""
    _require_sparsecore()
    p = _pick_partition(b, N_1M, k)
    assert p * min(k, N_1M // p) > MAX_SLICE_WORDS, (
        f"{b=} {k=} no longer reaches the multi-partition stage-2 path (p={p})"
    )
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, N_1M, rng)
    row_lengths = np.full(b, N_1M, np.int32)
    assert _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None


@pytest.mark.cpu_test
@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
def test_partition_factor_is_what_these_tests_assume(b, n, k, expected_p):
    """Pin `P` so a change to `_pick_partition` cannot silently retarget these."""
    assert _pick_partition(b, n, k) == expected_p


@pytest.mark.cpu_test
def test_select_p_and_env_override(monkeypatch):
    """Verify `_select_p` honors `max_p` and `TPU_SC_TOPK_MAX_P` while respecting `p_min`."""
    # Default at b=8, n=65536, k=2048: p_min=2, b*p reaches 32 at p=4.
    assert _select_p(8, 65536, 2048, None) == 4
    # Explicit max_p=2 clamps p from 4 down to p_min=2.
    assert _select_p(8, 65536, 2048, 2) == 2
    # Explicit max_p=1 cannot go below hardware SRAM floor p_min=2 for n=65536.
    assert _select_p(8, 65536, 2048, 1) == 2
    # For n=32768 (p_min=1), max_p=1 forces p=1 even at small batch b=4.
    assert _select_p(4, 32768, 2048, None) == 8
    assert _select_p(4, 32768, 2048, 1) == 1
    assert _select_p(4, 32768, 2048, 2) == 2
    # Environment variable override TPU_SC_TOPK_MAX_P.
    monkeypatch.setenv("TPU_SC_TOPK_MAX_P", "4")
    assert _select_p(4, 32768, 2048, None) == 4


@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
@pytest.mark.parametrize("kind", ["strict_desc", "random", "heavy_ties"])
def test_exact_topk_full_rows(b, n, k, expected_p, kind):
    _require_sparsecore()
    rng = np.random.default_rng(SEED)
    scores = _scores(kind, b, n, rng)
    row_lengths = np.full(b, n, np.int32)
    assert _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None


@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
def test_exact_topk_ragged_rows_straddle_partition_boundaries(b, n, k, expected_p):
    """Ragged lengths straddling subcore slice boundaries."""
    _require_sparsecore()
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
    _require_sparsecore()
    rng = np.random.default_rng(SEED)
    scores = _scores("all_neg_inf", b, n, rng)
    row_lengths = np.full(b, n, np.int32)
    idx = _run(scores, k, row_lengths)
    assert (idx == -1).all(), f"expected all -1, got {np.unique(idx)[:8]}"
    assert _mismatch(scores, idx, k, row_lengths) is None


def test_topk_concentrated_in_the_last_partition():
    """All mass sits in the final slice while earlier slices are `-1e30`."""
    _require_sparsecore()
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
    """The meta-test: prove `_mismatch` can reject bad output."""
    _require_sparsecore()
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
        (32, 2048, 256),  # p == 1 single_stage
        (4, 8192, 256),  # p == 8 VMEM_SHARED single_stage
        (64, N_1M, 2048),  # p == 32 two_stage
    ],
)
def test_return_scores(b: int, n: int, k: int):
    """Verifies that return_scores=True returns matching scores for winners and
    -inf for pad across p=1, VMEM_SHARED p>1, and two-stage p>16.
    """
    _require_sparsecore()
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


@pytest.mark.parametrize(
    "b,n,k,return_scores",
    [
        (32, 2048, 256, False),  # p == 1
        (4, 8192, 256, True),  # p == 8 VMEM_SHARED
        (8, 49152, 512, False),  # padded non-multiple of p*128
        (64, N_1M, 2048, False),  # p == 32 two-stage fallback
    ],
)
def test_values_payload_tracks_selected_indices(
    b: int, n: int, k: int, return_scores: bool
):
    """Verifies that `values=` payload returns `values[r, idx]` for winners and `-1` for pad."""
    _require_sparsecore()
    rng = np.random.default_rng(SEED + b + n)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    row_offs = (np.arange(b, dtype=np.int32)[:, None] + 1) * (n + 100)
    payload = row_offs + np.arange(n, dtype=np.int32)[None, :]
    row_lengths = rng.integers(k // 2, n + 1, size=(b,), dtype=np.int32)

    res = sparsecore_topk(
        jnp.asarray(scores),
        k,
        jnp.asarray(row_lengths),
        values=jnp.asarray(payload),
        return_scores=return_scores,
    )
    if return_scores:
        actual_vals, _ = res
    else:
        actual_vals = res

    actual_vals = np.asarray(actual_vals)
    actual_idxs = np.where(actual_vals >= 0, actual_vals - row_offs, -1)
    assert _mismatch(scores, actual_idxs, k, row_lengths) is None

    for r in range(b):
        live = actual_vals[r] >= 0
        np.testing.assert_array_equal(
            actual_vals[r, live], payload[r, actual_idxs[r, live]]
        )
        assert np.all(actual_vals[r, ~live] == -1)


# --------------------------------------------------------------------------- #
# Scheduling groups.
# --------------------------------------------------------------------------- #
# `b=64, n=N_1M` (`p=32 > 16`) is the two-stage shape; `b=64, n=N_SHORT` (`p=1`)
# and `b=16, n=N_128K` (`p=8` VMEM_SHARED) are single-stage shapes.
TWO_STAGE = (64, N_1M, 2048)
SINGLE_STAGE = (64, N_SHORT, 2048)
SINGLE_STAGE_VMEM_SHARED = (16, N_128K, 2048)


def _stage_group_ids(monkeypatch, b, n, k, **kwargs) -> list[int | None]:
    """The `scheduling_group_id` each `_sc_topk_direct` call receives, in order."""
    seen: list[int | None] = []

    def _stub(
        scores,
        k_,
        row_lengths,
        *,
        p=1,
        values=None,
        return_scores=True,
        write_empty=True,
        scheduling_group_id=None,
    ):
        seen.append(scheduling_group_id)
        out = jnp.zeros((scores.shape[0], k_), jnp.int32)
        return (out, out) if return_scores else out

    monkeypatch.setattr(sc_module, "_sc_topk_direct", _stub)
    sc_module.sparsecore_topk.__wrapped__(jnp.zeros((b, n), jnp.float32), k, **kwargs)
    return seen


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "shape,kwargs,expected",
    [
        # Two-stage (p=32 at 1M): each stage gets its own group.
        (TWO_STAGE, dict(scheduling_group_id=3, stage2_scheduling_group_id=4), [3, 4]),
        # Stage 2 unannotated: it must NOT inherit stage 1's group.
        (TWO_STAGE, dict(scheduling_group_id=3), [3, None]),
        # Stage 1 unannotated, stage 2 annotated: the two are independent.
        (TWO_STAGE, dict(stage2_scheduling_group_id=4), [None, 4]),
        (TWO_STAGE, {}, [None, None]),
        # Single-stage p=1 and p=8 VMEM_SHARED: stage-2 id is accepted and unused.
        (SINGLE_STAGE, dict(scheduling_group_id=3, stage2_scheduling_group_id=4), [3]),
        (SINGLE_STAGE, dict(stage2_scheduling_group_id=4), [None]),
        (
            SINGLE_STAGE_VMEM_SHARED,
            dict(scheduling_group_id=3, stage2_scheduling_group_id=4),
            [3],
        ),
    ],
    ids=[
        "both",
        "stage1_only",
        "stage2_only",
        "neither",
        "p1_both",
        "p1_stage2_only",
        "p8_vmem_shared_both",
    ],
)
def test_each_stage_gets_the_group_id_it_was_given(
    monkeypatch, shape, kwargs, expected
):
    b, n, k = shape
    assert _stage_group_ids(monkeypatch, b, n, k, **kwargs) == expected


@pytest.mark.cpu_test
def test_scheduling_group_ids_reach_the_lowered_hlo():
    """Verify both scheduling group IDs survive as frontend attributes in lowered HLO."""
    b, n, k = TWO_STAGE
    assert _pick_partition(b, n, k) == 32
    scores = jax.ShapeDtypeStruct((b, n), jnp.float32)
    row_lengths = jax.ShapeDtypeStruct((b,), jnp.int32)

    def _text(**kwargs):
        return _lower_to_tpu_text(sparsecore_topk, scores, k, row_lengths, **kwargs)

    both = _text(scheduling_group_id=3, stage2_scheduling_group_id=4)
    assert '_scheduling_group_id = "3"' in both
    assert '_scheduling_group_id = "4"' in both

    stage1_only = _text(scheduling_group_id=3)
    assert '_scheduling_group_id = "3"' in stage1_only
    assert '_scheduling_group_id = "4"' not in stage1_only
    assert "_scheduling_group_id" not in _text()


@pytest.mark.parametrize("kind", ["random", "heavy_ties"])
def test_scheduling_group_ids_do_not_change_the_selection(kind):
    """A scheduling hint that changed the answer would be a correctness bug."""
    _require_sparsecore()
    b, n, k = SINGLE_STAGE_VMEM_SHARED
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
    np.testing.assert_array_equal(
        np.sort(np.take_along_axis(scores, np.maximum(plain, 0), axis=1), axis=1),
        np.sort(np.take_along_axis(scores, np.maximum(grouped, 0), axis=1), axis=1),
    )


@pytest.mark.cpu_test
def test_zero_take_along_axis_and_zero_gather_in_lowered_hlo():
    """Verify 0 `take_along_axis` in kernel sources and 0 `stablehlo.gather` in lowered HLO."""
    kernel_root = Path(sc_module.__file__).resolve().parent
    target_files = [
        kernel_root / "sparsecore_topk.py",
        kernel_root / "streamindex_topk" / "dcp_sc_compact.py",
        kernel_root / "streamindex_topk" / "streamindex_topk.py",
    ]
    for path in target_files:
        src = path.read_text()
        assert "take_along_axis" not in src, f"Found take_along_axis in {path}"
        tree = ast.parse(src)
        attrs = [
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        ]
        assert "take_along_axis" not in attrs, f"Found AST take_along_axis in {path}"

    for m, n, k, with_val, ret_scores, max_p in [
        (32, 10240, 2048, False, True, None),  # P == 1 single-stage
        (8, 65536, 2048, True, True, 4),  # P == 4 VMEM_SHARED with payload + scores
        (8, 49152, 2048, True, False, None),  # padded non-multiple of P*128
        (64, N_1M, 2048, True, True, None),  # P == 32 two-stage fallback
    ]:
        x_spec = jax.ShapeDtypeStruct((m, n), jnp.float32)
        if with_val:
            v_spec = jax.ShapeDtypeStruct((m, n), jnp.int32)
            fn = lambda s, v: sparsecore_topk(
                s, k, values=v, return_scores=ret_scores, max_p=max_p
            )
            with _tpu_lowering_context():
                traced = jax.jit(fn).trace(x_spec, v_spec)
                hlo = traced.lower(lowering_platforms=("tpu",)).as_text()
        else:
            fn = lambda s: sparsecore_topk(
                s, k, return_scores=ret_scores, max_p=max_p
            )
            with _tpu_lowering_context():
                traced = jax.jit(fn).trace(x_spec)
                hlo = traced.lower(lowering_platforms=("tpu",)).as_text()
        assert "stablehlo.gather" not in hlo, (
            f"Found stablehlo.gather in lowered HLO for m={m}, n={n}, "
            f"with_val={with_val}, ret_scores={ret_scores}, max_p={max_p}"
        )
