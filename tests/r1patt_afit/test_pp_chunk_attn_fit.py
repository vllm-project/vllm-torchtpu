# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""TPU_PP_CHUNK_ATTN_FIT: the pipeline chunk scheduler's attention-slope fit.

The step cost model takes one attention slope per rank from step time
differences of a millisecond or two and, by default, keeps the largest of the
eight. On r1patth1 run02 one rank (PP3_PCP1) measured 5.0e-8 while its PCP0
twin measured 2.8e-8; the max fit then modelled a full 8,704-token step behind
the bench's 95,744-token prefix over target and the scheduler halved every step
(64 x 4,352 instead of 32 x 8,704, -30% full-step). ``median`` keeps the
median slope instead. Default ``max``: the flag-off model is the base15 model.

The profiles below are the real per-rank samples six engines logged (run01-03 of
two pods)
(ms rounded to 0.1 in the log, so a rebuilt slope is within ~2.5% of the
logged one). They also check the rebuild: the max fit must clip run02 and not
run01 / run03, as the bench observed.
"""

import math
import os
import statistics
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r1patt_afit_profiles import PROFILES  # noqa: E402

from vllm_torchtpu.core.pp_chunks import StepCostModel  # noqa: E402

SLACK = 0.7  # the A line's TPU_PP_CHUNK_SLACK
GRANULARITY = 5120  # the logged scheduler granularity (2,560 per rank x PCP 2)
PREFIX = 95744  # bench case 95744:8704:32
TOP = 8704


def _ranks(arm: str) -> list[list[tuple[int, int, float]]]:
    return [PROFILES[arm][k] for k in sorted(PROFILES[arm])]


def _synthetic(slopes: list[float]) -> list[list[tuple[int, int, float]]]:
    """One rank per slope: run01 PP0_PCP0's prefix-free ladder, its prefixed
    points rebuilt as ladder + slope * tokens * prefix (so each rank's fitted
    slope is exactly its slope)."""
    ref = PROFILES["A1"]["PP0_PCP0"]
    base = {t: ms for t, p, ms in ref if p == 0}
    ranks = []
    for s in slopes:
        ranks.append([(t, p, base[t] + (s * t * p if p else 0.0)) for t, p, _ in ref])
    return ranks


def _reference_fit_base15(stage_samples, slack, granularity, schedule=None):
    """base15 StepCostModel.fit, copied verbatim (arithmetic and order)."""
    ladders, slopes = [], []
    for samples in stage_samples:
        timed: dict[tuple[int, int], float] = {}
        for tokens, prefix, ms in samples:
            key = (int(tokens), int(prefix))
            timed[key] = min(timed.get(key, math.inf), float(ms))
        base = {t: ms for (t, p), ms in timed.items() if p == 0}
        stage_slopes = []
        for t in base:
            prefixed = sorted(
                (p, ms) for (tt, p), ms in timed.items() if tt == t and p > 0
            )
            if len(prefixed) >= 2:
                (p1, ms1), (p2, ms2) = prefixed[0], prefixed[-1]
                stage_slopes.append(max((ms2 - ms1) / (t * (p2 - p1)), 0.0))
            elif prefixed:
                p1, ms1 = prefixed[0]
                stage_slopes.append(max((ms1 - base[t]) / (t * p1), 0.0))
        attn = statistics.median(stage_slopes) if stage_slopes else 0.0
        ladders.append({t: max(ms - attn * t * t / 2, 0.0) for t, ms in base.items()})
        slopes.append(attn)
    buckets = sorted(ladders[0])
    linear = {b: max(ladder[b] for ladder in ladders) for b in buckets}
    attn = max(slopes)
    top = buckets[-1]
    full_step = linear[top] + attn * top * top / 2
    return buckets, linear, attn, full_step * (1.0 + slack), int(granularity), schedule, slopes


def _base15_describe(buckets, linear, attn, target, granularity) -> str:
    ladder = ", ".join(f"{b}: {linear[b]:.1f}" for b in buckets)
    return (
        f"linear ms per bucket {{{ladder}}}, attention {attn:.3e} ms per token "
        f"pair, target {target:.1f} ms, granularity {granularity} tokens"
    )


def _full_step_ms(model: StepCostModel, prefix: int = PREFIX) -> float:
    return model.step_ms(TOP, float(TOP) * prefix, float(TOP) * TOP)


def _first_chunk(model: StepCostModel, prefix: int = PREFIX) -> int:
    """The chunk an empty step gives a request with 8,704 tokens left."""
    return model.chunk(prefix, TOP, 0, 0.0, 0.0)


REAL = ["A1", "B1", "A2", "A3", "B2", "A4"]
CASES = {
    **{arm: _ranks(arm) for arm in REAL},
    "outlier": _synthetic([2.8e-8] * 7 + [5.0e-8]),
}


def test_env_default_is_max(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("TPU_PP_CHUNK_ATTN_FIT", raising=False)
    assert envs.TPU_PP_CHUNK_ATTN_FIT == "max"
    monkeypatch.setenv("TPU_PP_CHUNK_ATTN_FIT", " Median ")
    assert envs.TPU_PP_CHUNK_ATTN_FIT == "median"


@pytest.mark.parametrize("case", sorted(CASES))
def test_flag_off_fit_is_base15(case):
    samples = CASES[case]
    model = StepCostModel.fit(samples, SLACK, GRANULARITY)
    buckets, linear, attn, target, gran, schedule, _ = _reference_fit_base15(
        samples, SLACK, GRANULARITY
    )
    assert model.buckets == buckets
    assert model.linear_ms == linear
    assert model.attn_ms_per_token_pair == attn  # bitwise: same arithmetic
    assert model.target_ms == target
    assert (model.granularity, model.schedule) == (gran, schedule)
    assert (model.attn_fit, model.rank_attn) == ("max", ())
    assert model == StepCostModel.fit(samples, SLACK, GRANULARITY, attn_fit="max")
    assert model.describe() == _base15_describe(buckets, linear, attn, target, gran)


@pytest.mark.parametrize("arm", REAL)
def test_real_slopes_median_keeps_full_step(arm):
    samples = CASES[arm]
    med = StepCostModel.fit(samples, SLACK, GRANULARITY, attn_fit="median")
    mx = StepCostModel.fit(samples, SLACK, GRANULARITY)
    slopes = _reference_fit_base15(samples, SLACK, GRANULARITY)[-1]
    assert med.attn_ms_per_token_pair == statistics.median(slopes)
    assert med.rank_attn == tuple(slopes)
    step_med, step_max = _full_step_ms(med), _full_step_ms(mx)
    print(
        f"R1PATT_AFIT {arm}: max attn {mx.attn_ms_per_token_pair:.3e} target "
        f"{mx.target_ms:.2f} step {step_max:.2f} margin {mx.target_ms - step_max:+.2f} "
        f"chunk {_first_chunk(mx)} | median attn {med.attn_ms_per_token_pair:.3e} "
        f"target {med.target_ms:.2f} step {step_med:.2f} margin "
        f"{med.target_ms - step_med:+.2f} chunk {_first_chunk(med)}"
    )
    # The median fit models the bench's full step under target: no clip.
    assert step_med < med.target_ms
    assert _first_chunk(med) == TOP
    # The rebuilt max fit reproduces what the bench ran: both B engines
    # (run02 of two pods) clipped, the four A engines did not.
    clipped = arm.startswith("B")
    assert (step_max > mx.target_ms) is clipped
    assert (_first_chunk(mx) < TOP) is clipped


def test_one_outlier_rank_does_not_clip_with_median():
    samples = CASES["outlier"]
    med = StepCostModel.fit(samples, SLACK, GRANULARITY, attn_fit="median")
    mx = StepCostModel.fit(samples, SLACK, GRANULARITY)
    assert med.attn_ms_per_token_pair == pytest.approx(2.8e-8, rel=1e-9)
    assert mx.attn_ms_per_token_pair == pytest.approx(5.0e-8, rel=1e-9)
    print(
        f"R1PATT_AFIT outlier: max target {mx.target_ms:.2f} step {_full_step_ms(mx):.2f} "
        f"chunk {_first_chunk(mx)} | median target {med.target_ms:.2f} step "
        f"{_full_step_ms(med):.2f} chunk {_first_chunk(med)}"
    )
    assert _full_step_ms(med) < med.target_ms and _first_chunk(med) == TOP
    assert _full_step_ms(mx) > mx.target_ms and _first_chunk(mx) < TOP


def test_median_fit_names_itself_in_the_log_line():
    med = StepCostModel.fit(CASES["B1"], SLACK, GRANULARITY, attn_fit="median")
    text = med.describe()
    assert text.startswith(_base15_describe(
        med.buckets, med.linear_ms, med.attn_ms_per_token_pair, med.target_ms,
        med.granularity))
    assert ", attention fit median of 8 ranks (max 5.049e-08; per rank " in text


def test_unknown_fit_is_refused():
    with pytest.raises(ValueError, match="TPU_PP_CHUNK_ATTN_FIT"):
        StepCostModel.fit(CASES["A1"], SLACK, GRANULARITY, attn_fit="mean")


@pytest.mark.parametrize("flag", [None, "median"])
def test_install_reads_the_flag(monkeypatch, flag):
    """profile_and_install_cost_model hands the scheduler the fit the flag names."""
    from vllm_torchtpu.core import pp_chunk_scheduler as pcs

    if flag is None:
        monkeypatch.delenv("TPU_PP_CHUNK_ATTN_FIT", raising=False)
    else:
        monkeypatch.setenv("TPU_PP_CHUNK_ATTN_FIT", flag)
    installed = []
    sched = object.__new__(pcs.TpuPipelineChunkScheduler)
    monkeypatch.setattr(sched, "install_cost_model", installed.append, raising=False)
    monkeypatch.setenv("TPU_PP_CHUNK_SLACK", str(SLACK))

    class _Executor:
        def collective_rpc(self, name):
            assert name == "profile_pipeline_chunks"
            return [
                {"granularity": GRANULARITY, "samples": s, "schedule": None}
                for s in CASES["B1"]
            ]

    pcs.profile_and_install_cost_model(sched, _Executor())
    (model,) = installed
    want = StepCostModel.fit(CASES["B1"], SLACK, GRANULARITY, attn_fit=flag or "max")
    assert model == want
    assert (_first_chunk(model) == TOP) is (flag == "median")
