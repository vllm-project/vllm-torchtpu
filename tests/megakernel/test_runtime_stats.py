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
"""Step statistics tests.

Verifies batch-shape inspection from host metadata, the step-mix tracker, and
the A/B latency calibrator.
"""

from __future__ import annotations

import contextlib
import itertools
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm_torchtpu.megakernel.runtime.stats import (
    BatchShape,
    MKABCalibrator,
    StepMixTracker,
    check_mrope_uniform,
    check_qsl_uniform,
    inspect_batch_shape,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

EXPECTED_MEAN_DELTA_MS = -4.0


@contextlib.contextmanager
def _assert_raises(exc_type: type[BaseException], match: str) -> Iterator[None]:
    """Assert that the block raises `exc_type` containing `match`."""
    caught: BaseException | None = None
    try:
        yield
    except exc_type as exc:
        caught = exc
    if caught is None:
        msg = f"Expected {exc_type.__name__} matching {match!r}"
        raise AssertionError(msg)
    assert match in str(caught), f"Expected {match!r} in {caught!r}"


def _shape(
    *,
    eligible: bool,
    num_reqs: int,
    tokens_per_seq: int,
    reason: str,
) -> BatchShape:
    """Build a `BatchShape` using keyword-only fields."""
    return BatchShape(
        eligible=eligible,
        num_reqs=num_reqs,
        tokens_per_seq=tokens_per_seq,
        reason=reason,
    )


def test_qsl_and_mrope_uniformity() -> None:
    """Verify `check_qsl_uniform` and `check_mrope_uniform` across step sizes."""
    assert check_qsl_uniform([0, 1, 2, 3], 3) == _shape(
        eligible=True, num_reqs=3, tokens_per_seq=1, reason="ok"
    )
    assert check_qsl_uniform([0, 4, 8], 2) == _shape(
        eligible=True, num_reqs=2, tokens_per_seq=4, reason="ok"
    )
    assert check_qsl_uniform([0, 1, 5], 2) == _shape(
        eligible=False, num_reqs=2, tokens_per_seq=0, reason="stock_mixed"
    )
    assert check_qsl_uniform([0, 8], 1) == _shape(
        eligible=True, num_reqs=1, tokens_per_seq=8, reason="ok"
    )
    assert check_qsl_uniform([0, 16], 1) == _shape(
        eligible=True, num_reqs=1, tokens_per_seq=16, reason="ok"
    )
    assert check_qsl_uniform([0], 2) == _shape(
        eligible=False, num_reqs=0, tokens_per_seq=0, reason="stock_n0"
    )
    assert check_qsl_uniform([0, 0], 1) == _shape(
        eligible=False, num_reqs=1, tokens_per_seq=0, reason="stock_q0"
    )

    mrope_ok = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.int32)
    mrope_bad = np.array([[1, 2], [1, 3], [1, 2]], dtype=np.int32)
    assert check_mrope_uniform(mrope_ok, active_rows=2) is True
    assert check_mrope_uniform(mrope_bad, active_rows=2) is False


def test_inspect_batch_shape_runner_and_fallback_paths() -> None:
    """Verify runner and fallback paths in `inspect_batch_shape`."""
    bs_runner = inspect_batch_shape(
        num_reqs=2,
        has_roff=True,
        decode_and_total_reqs=(2, 2),
        qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_runner == _shape(eligible=True, num_reqs=2, tokens_per_seq=4, reason="ok")

    bs_n0 = inspect_batch_shape(
        num_reqs=0,
        has_roff=False,
        decode_and_total_reqs=None,
        qsl=np.array([0, 1, 2, 3], dtype=np.int32),
        fallback_decode_and_total_reqs=(3, 3),
        fallback_num_seqs=3,
        fallback_qsl=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert bs_n0 == _shape(
        eligible=False, num_reqs=0, tokens_per_seq=0, reason="stock_n0"
    )

    bs_fb = inspect_batch_shape(
        num_reqs=None,
        has_roff=False,
        decode_and_total_reqs=None,
        qsl=None,
        fallback_decode_and_total_reqs=(3, 3),
        fallback_num_seqs=3,
        fallback_qsl=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert bs_fb == _shape(eligible=True, num_reqs=3, tokens_per_seq=1, reason="ok")

    bs_no_qsl = inspect_batch_shape(
        num_reqs=3,
        has_roff=False,
        decode_and_total_reqs=(3, 3),
        qsl=None,
    )
    assert bs_no_qsl == _shape(
        eligible=False, num_reqs=0, tokens_per_seq=0, reason="stock_no_qsl"
    )

    bs_no_roff_q4 = inspect_batch_shape(
        num_reqs=2,
        has_roff=False,
        qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_no_roff_q4 == _shape(
        eligible=False, num_reqs=2, tokens_per_seq=0, reason="stock_prefill"
    )
    bs_fb_q4 = inspect_batch_shape(
        num_reqs=None,
        has_roff=False,
        qsl=None,
        fallback_num_seqs=2,
        fallback_qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_fb_q4 == _shape(eligible=True, num_reqs=2, tokens_per_seq=4, reason="ok")

    untyped_inspect: Callable[..., object] = inspect_batch_shape
    with _assert_raises(TypeError, "unexpected keyword argument"):
        untyped_inspect(qsl_host=[0, 1])
    with _assert_raises(TypeError, "unexpected keyword argument"):
        untyped_inspect(comb_windowed_and_total=(1, 1))


def test_step_mix_tracker_and_ab_calibrator() -> None:
    """Verify `StepMixTracker` and `MKABCalibrator` alternation and quantiles."""
    tracker = StepMixTracker(period_s=0.0)
    run_ok, snap = tracker.note_step("mk_t1_b8", run=True)
    assert run_ok is True
    assert snap is not None
    assert snap[1] == 1
    assert snap[4] == {"mk_t1_b8": 1}

    ab = MKABCalibrator(enabled=True, period_s=0.0, cap=3)
    run0, k0, d0 = ab.choose("mk_t1_b8", mk_eligible=True)
    assert (run0, k0, d0) == (True, "mk_t1_b8_ab_mk", "mk")
    res = ab.record(1.0, padded_tokens=8)
    assert res is not None
    assert res[0] >= 0.0
    win, cum = json.loads(res[1]), json.loads(res[2])
    assert win == cum == {"mk_t1_b8_p8": {"mk": [1, 1.0, 1.0, 1.0, 1.0]}}
    run1, k1, d1 = ab.choose("mk_t1_b8", mk_eligible=True)
    assert (run1, k1, d1) == (False, "mk_t1_b8_ab_stock", "stock")
    res = ab.record(5.0, padded_tokens=8)
    assert res is not None
    win, cum = json.loads(res[1]), json.loads(res[2])
    assert win == {"mk_t1_b8_p8": {"stock": [1, 5.0, 5.0, 5.0, 5.0]}}
    assert cum["mk_t1_b8_p8"]["d_mean_ms"] == EXPECTED_MEAN_DELTA_MS
    for mk_ms, stock_ms in ((2.0, 6.0), (3.0, 7.0), (10.0, 8.0)):
        for arm, dt in (("mk", mk_ms), ("stock", stock_ms)):
            assert ab.choose("mk_t1_b8", mk_eligible=True)[2] == arm
            res = ab.record(dt, padded_tokens=8)
    assert res is not None
    win, cum = json.loads(res[1]), json.loads(res[2])
    assert win == {"mk_t1_b8_p8": {"stock": [1, 8.0, 8.0, 8.0, 8.0]}}
    assert cum == {
        "mk_t1_b8_p8": {
            "mk": [4, 4.0, 2.2, 3.0, 8.6],
            "stock": [4, 6.5, 6.2, 7.0, 7.8],
            "d_mean_ms": -2.5,
            "d_p50_ms": -4.0,
        }
    }
    ab.choose("mk_t4_b16", mk_eligible=True)
    res = ab.record(2.0)
    assert res is not None
    assert "mk_t4_b16" in json.loads(res[2])
    slow = MKABCalibrator(enabled=True, period_s=3600.0)
    slow.choose("mk_t1_b8", mk_eligible=True)
    assert slow.record(1.0, padded_tokens=8) is None
    off = MKABCalibrator(enabled=False, period_s=0.0)
    assert off.choose("mk_t1_b8", mk_eligible=True) == (
        True,
        "mk_t1_b8",
        "mk",
    )
    assert off.choose("stock_big", mk_eligible=False) == (
        False,
        "stock_big",
        "stock",
    )
    assert off.record(1.0, padded_tokens=8) is None


@dataclass(frozen=True)
class _RefCase:
    """Input bundle for `_reference_batch_shape`."""

    num_reqs: int | None
    has_roff: bool
    comb_wt: tuple[int, int] | None
    qsl_host: list[int] | None
    dist_dt: tuple[int, int] | None
    fb_nseq: int | None
    qsl_md: list[int] | None


def _reference_runner_gate(
    case: _RefCase,
) -> tuple[int | None, BatchShape | None]:
    """Evaluate runner `decode_and_total_reqs` and `qsl_host` rules."""
    num_reqs = case.num_reqs
    if case.has_roff and case.comb_wt is not None:
        num_windowed, dist_reqs = int(case.comb_wt[0]), int(case.comb_wt[1])
        if num_reqs is None:
            num_reqs = dist_reqs
        if dist_reqs <= 0 or num_reqs <= 0:
            return num_reqs, _shape(
                eligible=False, num_reqs=0, tokens_per_seq=0, reason="stock_n0"
            )
        if num_windowed != dist_reqs:
            reason = "stock_prefill" if num_windowed == 0 else "stock_mixed"
            return num_reqs, _shape(
                eligible=False,
                num_reqs=num_reqs,
                tokens_per_seq=0,
                reason=reason,
            )
    if case.qsl_host is not None and num_reqs is not None:
        bs = check_qsl_uniform(case.qsl_host, num_reqs)
        if bs.eligible and not case.has_roff and bs.tokens_per_seq != 1:
            return num_reqs, _shape(
                eligible=False,
                num_reqs=bs.num_reqs,
                tokens_per_seq=0,
                reason="stock_prefill",
            )
        return num_reqs, bs
    return num_reqs, None


def _reference_batch_shape(case: _RefCase) -> BatchShape:
    """Evaluate independent straight-line statement of the batch-shape rules."""
    num_reqs, early = _reference_runner_gate(case)
    if early is not None:
        return early
    if case.dist_dt is not None:
        d_end, total_reqs = int(case.dist_dt[0]), int(case.dist_dt[1])
        if total_reqs <= 0:
            return _shape(
                eligible=False, num_reqs=0, tokens_per_seq=0, reason="stock_n0"
            )
        if d_end != total_reqs:
            reason = "stock_prefill" if d_end == 0 else "stock_mixed"
            return _shape(
                eligible=False,
                num_reqs=total_reqs,
                tokens_per_seq=0,
                reason=reason,
            )
        if num_reqs is None:
            num_reqs = total_reqs
    if num_reqs is None and case.fb_nseq is not None:
        num_reqs = int(case.fb_nseq)
    if case.qsl_md is not None and num_reqs is not None:
        return check_qsl_uniform(case.qsl_md, num_reqs)
    return _shape(eligible=False, num_reqs=0, tokens_per_seq=0, reason="stock_no_qsl")


def test_inspect_batch_shape_matches_the_reference_exhaustively() -> None:
    """Verify `inspect_batch_shape` matches `_reference_batch_shape` exhaustively."""
    num_reqs_vals = (None, -1, 0, 1, 2)
    has_roff_vals = (False, True)
    comb_vals = (None, (0, 0), (1, 0), (0, 2), (1, 2), (2, 2))
    qsl_vals = (
        None,
        [0],
        [0, 0],
        [0, 1, 2],
        [0, 4, 8],
        [0, 1, 5],
        [0, 8, 16],
    )
    dist_vals = (None, (0, 0), (1, 2), (2, 2))
    fb_nseq_vals = (None, 0, 1, 2)
    qsl_md_vals = (None, [0], [0, 1, 2], [0, 4, 8], [0, 1, 5])

    total_checked = 0
    for nr, hr, cw, qh, dt, fn, qm in itertools.product(
        num_reqs_vals,
        has_roff_vals,
        comb_vals,
        qsl_vals,
        dist_vals,
        fb_nseq_vals,
        qsl_md_vals,
    ):
        total_checked += 1
        got = inspect_batch_shape(
            num_reqs=nr,
            has_roff=hr,
            decode_and_total_reqs=cw,
            qsl=qh,
            fallback_decode_and_total_reqs=dt,
            fallback_num_seqs=fn,
            fallback_qsl=qm,
        )
        case = _RefCase(nr, hr, cw, qh, dt, fn, qm)
        exp = _reference_batch_shape(case)
        assert got == exp, f"mismatch for {case}: got={got} vs exp={exp}"
    assert total_checked == 5 * 2 * 6 * 7 * 4 * 4 * 5
