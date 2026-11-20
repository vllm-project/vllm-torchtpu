# SPDX-License-Identifier: Apache-2.0
"""PrefillSpacingScheduler TPOT hold: priced bins, hold horizon, max wait.

The scheduler is built with ``__new__`` and only the state the tested methods
read, so no engine or KV cache manager is needed.
"""

import time
from types import SimpleNamespace

import pytest

from vllm_torchtpu.core import prefill_spacing_scheduler as pss
from vllm_torchtpu.core.prefill_spacing_scheduler import (
    _DECODE,
    _FIT_BINS,
    _LONG,
    PrefillSpacingScheduler,
)


def _sched(target_ms=6.0, horizon=1.0, priced=False, max_wait_s=0.0):
    s = PrefillSpacingScheduler.__new__(PrefillSpacingScheduler)
    s.tpot_target_s = target_ms / 1e3
    s.hold_horizon = horizon
    s.priced = priced
    s.max_wait_s = max_wait_s
    s._decode_t0 = {}
    s._tokens_per_step = 1.0
    s._slack_s = float("inf")
    s._step_s = {}
    s._bin_s = {}
    s.running = []
    return s


def _req(rid, max_tokens, out_tokens, **kw):
    return SimpleNamespace(
        request_id=rid, max_tokens=max_tokens, num_output_tokens=out_tokens, **kw
    )


def _old_short_of_slack(s, decoders, now):
    """The rule before the horizon: any request with 0 < slack < extra."""
    decode_s, long_s = s._step_s.get(_DECODE), s._step_s.get(_LONG)
    if decode_s is None or long_s is None:
        return False
    extra_s = long_s - decode_s
    for r in decoders:
        t0 = s._decode_t0.get(r.request_id)
        if t0 is None or r.max_tokens <= 1:
            continue
        steps_left = max(r.max_tokens - r.num_output_tokens, 0) / max(
            s._tokens_per_step, 1.0
        )
        slack_s = s.tpot_target_s * (r.max_tokens - 1) - (
            now - t0 + steps_left * decode_s
        )
        if 0 < slack_s < extra_s:
            return True
    return False


@pytest.mark.parametrize("elapsed", [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
@pytest.mark.parametrize("long_s", [0.01, 0.2, 0.4, 2.0])
def test_horizon_one_matches_previous_rule(elapsed, long_s):
    s = _sched()
    s._step_s = {_DECODE: 0.005, _LONG: long_s}
    now = 100.0
    decoders = [_req("a", 1000, 300), _req("b", 500, 10), _req("c", 1, 0)]
    s._decode_t0 = {"a": now - elapsed, "b": now - elapsed / 3}
    assert s._short_of_slack(decoders, now) == _old_short_of_slack(s, decoders, now)


def test_no_step_times_never_holds():
    s = _sched()
    assert not s._short_of_slack([_req("a", 100, 1)], 1.0)
    assert s._slack_s == float("inf")


def test_horizon_widens_hold():
    now = 50.0
    s = _sched(target_ms=6.0, horizon=1.0)
    s._step_s = {_DECODE: 0.005, _LONG: 0.305}
    # slack = 0.006 * 999 - (elapsed + 700 * 0.005) = 5.994 - 3.5 - elapsed
    s._decode_t0 = {"a": now - 2.1}
    dec = [_req("a", 1000, 300)]
    assert not s._short_of_slack(dec, now)  # slack 0.394 > 0.3
    assert s._slack_s == pytest.approx(0.394, abs=1e-6)
    s.hold_horizon = 1.5
    assert s._short_of_slack(dec, now)  # 0.394 < 0.45


def test_fit_tokens_picks_largest_fitting_bin():
    s = _sched(priced=True)
    s._step_s = {_DECODE: 0.005}
    s._bin_s = {512: 0.03, 1024: 0.06, 2048: 0.11, 4096: 0.25}
    s._slack_s = 0.1
    assert s._fit_tokens() == 1024
    s._slack_s = 0.01
    assert s._fit_tokens() == 0
    s._slack_s = 1.0
    assert s._fit_tokens() == 4096
    s._bin_s = {2048: 0.05}  # unmeasured bins never fit
    s._slack_s = 0.1
    assert s._fit_tokens() == 2048
    assert _FIT_BINS == (512, 1024, 2048, 4096)


def test_time_step_ema_and_outlier():
    avg = {}
    PrefillSpacingScheduler._time_step(avg, "k", 0.1)
    assert avg["k"] == 0.1
    PrefillSpacingScheduler._time_step(avg, "k", 0.2)
    assert avg["k"] == pytest.approx(0.1 + pss._STEP_TIME_EMA * 0.1)
    before = avg["k"]
    PrefillSpacingScheduler._time_step(avg, "k", 1e6)  # outlier ignored
    assert avg["k"] == before


def test_waited_too_long():
    s = _sched(max_wait_s=8.0)
    s._is_blocked_waiting_status = lambda status: status == "blocked"
    now = time.time()
    fresh = _req("w1", 10, 0, arrival_time=now - 1, status="waiting")
    old_blocked = _req("w2", 10, 0, arrival_time=now - 100, status="blocked")
    assert not s._waited_too_long([fresh, old_blocked])
    old = _req("w3", 10, 0, arrival_time=now - 9, status="waiting")
    assert s._waited_too_long([fresh, old])
    # A running prefill chunk counts; a request with output tokens does not.
    chunk = _req("r1", 10, 0, arrival_time=now - 20, is_prefill_chunk=True)
    s.running = [chunk]
    assert s._waited_too_long([])
    chunk.num_output_tokens = 1
    assert not s._waited_too_long(None)


def test_new_envs_default_off(monkeypatch):
    from vllm_torchtpu import envs

    for k in (
        "VLLM_TPU_PREFILL_TPOT_PRICED",
        "VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON",
        "VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S",
    ):
        monkeypatch.delenv(k, raising=False)
    assert envs.VLLM_TPU_PREFILL_TPOT_PRICED is False
    assert envs.VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON == 1.0
    assert envs.VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S == 0.0
