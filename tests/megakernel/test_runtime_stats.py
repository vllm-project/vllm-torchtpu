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
"""Step statistics: batch-shape inspection from host metadata, the step-mix
tracker and the A/B latency calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_torchtpu.megakernel.runtime.stats import (
    BatchShape,
    MKABCalibrator,
    StepMixTracker,
    check_mrope_uniform,
    check_qsl_uniform,
    inspect_batch_shape,
)


def test_qsl_and_mrope_uniformity() -> None:
    assert check_qsl_uniform([0, 1, 2, 3], 3) == BatchShape(True, 3, 1, "ok")
    assert check_qsl_uniform([0, 4, 8], 2) == BatchShape(True, 2, 4, "ok")
    assert check_qsl_uniform([0, 1, 5], 2) == BatchShape(False, 2, 0, "stock_mixed")
    assert check_qsl_uniform([0, 8], 1) == BatchShape(False, 1, 0, "stock_q8")
    assert check_qsl_uniform([0], 2) == BatchShape(False, 0, 0, "stock_n0")
    assert check_qsl_uniform([0, 0], 1) == BatchShape(False, 1, 0, "stock_q0")

    mrope_ok = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.int32)
    mrope_bad = np.array([[1, 2], [1, 3], [1, 2]], dtype=np.int32)
    assert check_mrope_uniform(mrope_ok, 2) is True
    assert check_mrope_uniform(mrope_bad, 2) is False


def test_inspect_batch_shape_runner_and_fallback_paths() -> None:
    # Runner path vs fallback path in inspect_batch_shape
    bs_runner = inspect_batch_shape(
        num_reqs=2,
        has_roff=True,
        decode_and_total_reqs=(2, 2),
        qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_runner == BatchShape(True, 2, 4, "ok")

    # num_reqs=0 is not None -> fails closed with stock_n0 (no fallback_num_seqs)
    bs_n0 = inspect_batch_shape(
        num_reqs=0,
        has_roff=False,
        decode_and_total_reqs=None,
        qsl=np.array([0, 1, 2, 3], dtype=np.int32),
        fallback_decode_and_total_reqs=(3, 3),
        fallback_num_seqs=3,
        fallback_qsl=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert bs_n0 == BatchShape(False, 0, 0, "stock_n0")

    # Fallback path when num_reqs is None and runner qsl/dist are None
    bs_fb = inspect_batch_shape(
        num_reqs=None,
        has_roff=False,
        decode_and_total_reqs=None,
        qsl=None,
        fallback_decode_and_total_reqs=(3, 3),
        fallback_num_seqs=3,
        fallback_qsl=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert bs_fb == BatchShape(True, 3, 1, "ok")

    # Missing both qsl and fallback_qsl fails closed with stock_no_qsl
    bs_no_qsl = inspect_batch_shape(
        num_reqs=3,
        has_roff=False,
        decode_and_total_reqs=(3, 3),
        qsl=None,
    )
    assert bs_no_qsl == BatchShape(False, 0, 0, "stock_no_qsl")

    # Primary qsl without roff rejects T=4 (stock_prefill); fallback_qsl allows T=4
    bs_no_roff_q4 = inspect_batch_shape(
        num_reqs=2,
        has_roff=False,
        qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_no_roff_q4 == BatchShape(False, 2, 0, "stock_prefill")
    bs_fb_q4 = inspect_batch_shape(
        num_reqs=None,
        has_roff=False,
        qsl=None,
        fallback_num_seqs=2,
        fallback_qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_fb_q4 == BatchShape(True, 2, 4, "ok")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        inspect_batch_shape(qsl_host=[0, 1])  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        inspect_batch_shape(comb_windowed_and_total=(1, 1))  # type: ignore[call-arg]


def test_step_mix_tracker_and_ab_calibrator() -> None:
    tracker = StepMixTracker(period_s=0.0)
    run_ok, snap = tracker.note_step("mk_t1_b8", True)
    assert run_ok is True
    assert snap is not None and snap[1] == 1 and snap[4] == {"mk_t1_b8": 1}

    ab = MKABCalibrator(enabled=True, period_s=60.0)
    run0, k0, d0 = ab.choose("mk_t1_b8", True)
    run1, k1, d1 = ab.choose("mk_t1_b8", True)
    assert (run0, k0, d0) == (True, "mk_t1_b8_ab_mk", "mk")
    assert (run1, k1, d1) == (False, "mk_t1_b8_ab_stock", "stock")
    ab.record(1.2, padded_tokens=8)


def _reference_batch_shape(
    num_reqs: int | None,
    has_roff: bool,
    comb_wt: tuple[int, int] | None,
    qsl_host: list[int] | None,
    dist_dt: tuple[int, int] | None,
    fb_nseq: int | None,
    qsl_md: list[int] | None,
) -> BatchShape:
    """An independent, straight-line statement of the batch-shape rules."""
    if has_roff and comb_wt is not None:
        num_windowed, dist_reqs = int(comb_wt[0]), int(comb_wt[1])
        if num_reqs is None:
            num_reqs = dist_reqs
        if dist_reqs <= 0 or num_reqs <= 0:
            return BatchShape(False, 0, 0, "stock_n0")
        if num_windowed != dist_reqs:
            return BatchShape(
                False,
                num_reqs,
                0,
                "stock_prefill" if num_windowed == 0 else "stock_mixed",
            )
    if qsl_host is not None and num_reqs is not None:
        bs = check_qsl_uniform(qsl_host, num_reqs)
        if bs.eligible and not has_roff and bs.tokens_per_seq != 1:
            return BatchShape(False, bs.num_reqs, 0, "stock_prefill")
        return bs

    if dist_dt is not None:
        d_end, total_reqs = int(dist_dt[0]), int(dist_dt[1])
        if total_reqs <= 0:
            return BatchShape(False, 0, 0, "stock_n0")
        if d_end != total_reqs:
            return BatchShape(
                False,
                total_reqs,
                0,
                "stock_prefill" if d_end == 0 else "stock_mixed",
            )
        if num_reqs is None:
            num_reqs = total_reqs
    if num_reqs is None and fb_nseq is not None:
        num_reqs = int(fb_nseq)
    if qsl_md is not None and num_reqs is not None:
        return check_qsl_uniform(qsl_md, num_reqs)
    return BatchShape(False, 0, 0, "stock_no_qsl")


def test_inspect_batch_shape_matches_the_reference_exhaustively() -> None:
    """`inspect_batch_shape` agrees with `_reference_batch_shape` on every
    combination of the inputs below."""
    num_reqs_vals = (None, -1, 0, 1, 2)
    has_roff_vals = (False, True)
    comb_vals = (None, (0, 0), (1, 0), (0, 2), (1, 2), (2, 2))
    qsl_vals = (None, [0], [0, 0], [0, 1, 2], [0, 4, 8], [0, 1, 5], [0, 8, 16])
    dist_vals = (None, (0, 0), (1, 2), (2, 2))
    fb_nseq_vals = (None, 0, 1, 2)
    qsl_md_vals = (None, [0], [0, 1, 2], [0, 4, 8], [0, 1, 5])

    total_checked = 0
    for nr in num_reqs_vals:
        for hr in has_roff_vals:
            for cw in comb_vals:
                for qh in qsl_vals:
                    for dt in dist_vals:
                        for fn in fb_nseq_vals:
                            for qm in qsl_md_vals:
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
                                exp = _reference_batch_shape(nr, hr, cw, qh, dt, fn, qm)
                                assert got == exp, (
                                    f"mismatch for {(nr, hr, cw, qh, dt, fn, qm)}: "
                                    f"got={got} vs exp={exp}"
                                )
    assert total_checked == 5 * 2 * 6 * 7 * 4 * 4 * 5
