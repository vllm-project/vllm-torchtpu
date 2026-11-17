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
"""PrefillSpacingScheduler's priced TPOT target (VLLM_TPU_PREFILL_TPOT_PRICE_RESERVE)."""

import time

import pytest
from vllm import SamplingParams
from vllm.v1.request import Request

from vllm_torchtpu.core.prefill_spacing_scheduler import (
    _DECODE,
    _LONG,
    PrefillSpacingScheduler,
)

from test_srpt_prefill_scheduler import (  # noqa: F401
    _build_real_scheduler,
    shared_model_config,
)

TARGET_MS = 10.0
DECODE_S = 0.005
MAX_TOKENS = 100


def _env(monkeypatch, priced: bool, min_chunk: int = 16, us_per_token: float = 40.0):
    monkeypatch.setenv("VLLM_TPU_PREFILL_SPACING_FRAC", "0")
    monkeypatch.setenv("VLLM_TPU_PREFILL_TPOT_TARGET_MS", str(TARGET_MS))
    monkeypatch.setenv("VLLM_TPU_PREFILL_TPOT_PRICE_RESERVE", "1e-9" if priced else "0")
    monkeypatch.setenv("VLLM_TPU_PREFILL_TPOT_MIN_CHUNK", str(min_chunk))
    monkeypatch.setenv("VLLM_TPU_PREFILL_TPOT_US_PER_TOKEN", str(us_per_token))


def _request(request_id, num_tokens, hasher, seed):
    params = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = [seed * 100_000 + i for i in range(num_tokens)]
    return Request(request_id, prompt, params, None, block_hasher=hasher)


def _decoding_with_slack(model_config, slack_s: float, budget: int = 64):
    """A scheduler with one decoding request whose TPOT slack is `slack_s`
    and a 200-token prompt waiting."""
    scheduler, hasher = _build_real_scheduler(
        model_config,
        scheduler_cls=PrefillSpacingScheduler,
        num_blocks=64,
        max_num_batched_tokens=budget,
    )
    dec = _request("dec", 16, hasher, seed=1)
    scheduler.add_request(dec)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"dec": 16}
    assert not dec.is_prefill_chunk
    scheduler._step_s = {_DECODE: DECODE_S, _LONG: 0.5}
    # projected = elapsed + steps_left * decode_s; slack = target * 99 - that.
    steps_left_s = MAX_TOKENS * DECODE_S
    elapsed = TARGET_MS / 1e3 * (MAX_TOKENS - 1) - steps_left_s - slack_s
    scheduler._decode_t0["dec"] = time.monotonic() - elapsed
    scheduler._seen_out["dec"] = dec.num_output_tokens
    scheduler.add_request(_request("long", 200, hasher, seed=2))
    return scheduler


def test_priced_caps_prefill_to_slack(shared_model_config, monkeypatch):
    _env(monkeypatch, priced=True)
    # 2 ms of slack at 40 us/token -> 50 prefill tokens (the 64 budget would
    # give the prompt 63).
    scheduler = _decoding_with_slack(shared_model_config, slack_s=0.002)
    out = scheduler.schedule()
    assert "dec" in out.num_scheduled_tokens
    assert 45 <= out.num_scheduled_tokens["long"] <= 50
    assert scheduler._capped_steps == 1


def test_unpriced_target_holds_all_prefill(shared_model_config, monkeypatch):
    _env(monkeypatch, priced=False)
    scheduler = _decoding_with_slack(shared_model_config, slack_s=0.002)
    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens) == {"dec"}


def test_priced_holds_below_min_chunk(shared_model_config, monkeypatch):
    _env(monkeypatch, priced=True, min_chunk=64)
    scheduler = _decoding_with_slack(shared_model_config, slack_s=0.002)
    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens) == {"dec"}


def test_priced_no_cap_with_enough_slack(shared_model_config, monkeypatch):
    _env(monkeypatch, priced=True)
    scheduler = _decoding_with_slack(shared_model_config, slack_s=0.1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["long"] == 64 - out.num_scheduled_tokens["dec"]
    assert scheduler._capped_steps == 0


def test_priced_ignores_requests_past_target(shared_model_config, monkeypatch):
    _env(monkeypatch, priced=True)
    scheduler = _decoding_with_slack(shared_model_config, slack_s=-0.05)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["long"] == 64 - out.num_scheduled_tokens["dec"]


def test_priced_learns_per_token_cost(shared_model_config, monkeypatch):
    _env(monkeypatch, priced=True, us_per_token=40.0)
    scheduler = _decoding_with_slack(shared_model_config, slack_s=0.1)
    # 1024 prefill tokens took 30 ms over a 5 ms decode step: 24.4 us/token.
    for _ in range(100):
        scheduler._learn_price(DECODE_S + 0.025, 1024)
    assert scheduler._s_per_token * 1e6 == pytest.approx(25e3 / 1024, rel=1e-3)
    # Small prefill steps do not move it.
    scheduler._learn_price(1.0, 100)
    assert scheduler._s_per_token * 1e6 == pytest.approx(25e3 / 1024, rel=1e-3)
    # One outlier moves it at most a bounded step.
    before = scheduler._s_per_token
    scheduler._learn_price(100.0, 1024)
    assert scheduler._s_per_token <= before * (1 + 0.1 * 3) + 1e-12


def test_priced_off_without_target(shared_model_config, monkeypatch):
    _env(monkeypatch, priced=True)
    monkeypatch.setenv("VLLM_TPU_PREFILL_TPOT_TARGET_MS", "0")
    scheduler, _ = _build_real_scheduler(
        shared_model_config, scheduler_cls=PrefillSpacingScheduler
    )
    assert scheduler.priced is False
