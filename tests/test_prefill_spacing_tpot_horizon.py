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
"""PrefillSpacingScheduler TPOT hold horizon and max wait."""

import time
from types import SimpleNamespace

import pytest

from vllm_torchtpu.core.prefill_spacing_scheduler import (
    _DECODE,
    _LONG,
    PrefillSpacingScheduler,
)


def _req(rid, out, max_tokens=1000, prefill=False, arrival=None):
    return SimpleNamespace(
        request_id=rid,
        num_output_tokens=out,
        max_tokens=max_tokens,
        is_prefill_chunk=prefill,
        arrival_time=time.time() if arrival is None else arrival,
    )


def _sched(horizon=0.0, max_wait=0.0, target_ms=6.0):
    s = object.__new__(PrefillSpacingScheduler)
    s.tpot_target_s = target_ms / 1e3
    s.max_hold_s = 10.0
    s.hold_horizon_s = horizon
    s.max_wait_s = max_wait
    s._wait_releases = 0
    s._decode_t0 = {}
    s._seen_out = {}
    s._tokens_per_step = 4.0
    s._held_since = None
    # 10 ms decode step, 400 ms long-chunk step: extra 0.39 s.
    s._step_s = {_DECODE: 0.010, _LONG: 0.400}
    s.running = []
    return s


def _in_band(s, r, now, slack_s=0.2):
    """Back-date the request's first decode so its slack is ``slack_s``."""
    steps_left = (r.max_tokens - r.num_output_tokens) / s._tokens_per_step
    projected_wo_elapsed = steps_left * s._step_s[_DECODE]
    elapsed = s.tpot_target_s * (r.max_tokens - 1) - projected_wo_elapsed - slack_s
    s._decode_t0[r.request_id] = now - elapsed
    s._seen_out[r.request_id] = r.num_output_tokens


@pytest.mark.parametrize("horizon,out,expect", [
    (0.0, 100, True),  # off: protected wherever it is
    (1.5, 100, False),  # 900 tokens left = 2.25 s > 1.5 s horizon
    (1.5, 500, True),  # 500 tokens left = 1.25 s <= 1.5 s
])
def test_horizon(horizon, out, expect):
    s = _sched(horizon=horizon)
    r = _req("a", out)
    now = time.monotonic()
    _in_band(s, r, now)
    assert s._short_of_slack([r], now) is expect


def test_horizon_keeps_band_rule():
    s = _sched(horizon=1.5)
    r = _req("a", 500)
    now = time.monotonic()
    _in_band(s, r, now, slack_s=0.5)  # slack above one long-step extra
    assert s._short_of_slack([r], now) is False
    _in_band(s, r, now, slack_s=-0.1)  # already past the target
    assert s._short_of_slack([r], now) is False


def test_max_wait_releases_hold():
    s = _sched(max_wait=8.0)
    r = _req("a", 500)
    now = time.monotonic()
    _in_band(s, r, now)
    young = [_req("w", 0, arrival=time.time() - 1.0)]
    assert s._hold_for_tpot([r], young) is True
    _in_band(s, r, time.monotonic())
    old = [_req("w", 0, arrival=time.time() - 9.0)]
    assert s._hold_for_tpot([r], old) is False
    assert s._wait_releases == 1
    assert s._held_since is None


def test_max_wait_counts_running_prefill_chunk():
    s = _sched(max_wait=8.0)
    r = _req("a", 500)
    s.running = [r, _req("p", 0, prefill=True, arrival=time.time() - 20.0)]
    now = time.monotonic()
    _in_band(s, r, now)
    # Nothing waiting, but the running chunk is pending prefill and old.
    assert s._hold_for_tpot([r], []) is False


def test_max_wait_off_holds():
    s = _sched(max_wait=0.0)
    r = _req("a", 500)
    _in_band(s, r, time.monotonic())
    old = [_req("w", 0, arrival=time.time() - 100.0)]
    assert s._hold_for_tpot([r], old) is True
