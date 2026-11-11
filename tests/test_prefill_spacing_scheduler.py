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
"""PrefillSpacingScheduler tests on a real scheduler, through schedule()."""

from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request

from tests.test_srpt_prefill_scheduler import _build_real_scheduler
from vllm_torchtpu.core import prefill_spacing_scheduler as pss
from vllm_torchtpu.core.prefill_spacing_scheduler import PrefillSpacingScheduler
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _request(request_id: str, num_tokens: int, hasher, seed: int) -> Request:
    params = SamplingParams(max_tokens=256, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = [seed * 100_000 + i for i in range(num_tokens)]
    return Request(request_id, prompt, params, None, block_hasher=hasher)


def _scheduler(model_config, scheduler_cls=PrefillSpacingScheduler, **kwargs):
    kwargs.setdefault("num_blocks", 64)
    return _build_real_scheduler(model_config, scheduler_cls=scheduler_cls, **kwargs)


@pytest.mark.parametrize(
    ("scheduler_cls", "expected"),
    [
        # Upstream runs the running chunk first; it takes the whole budget.
        (PrefixAwareScheduler, {"long": 32}),
        # The shorter waiting prompt is prefilled first, the chunk takes the rest.
        (PrefillSpacingScheduler, {"short": 16, "long": 16}),
    ],
)
def test_shorter_waiting_prompt_goes_before_running_chunk(
    model_config, scheduler_cls, expected
):
    scheduler, hasher = _scheduler(
        model_config, scheduler_cls, max_num_batched_tokens=32
    )
    scheduler.add_request(_request("long", 160, hasher, seed=1))
    assert scheduler.schedule().num_scheduled_tokens == {"long": 32}
    scheduler.add_request(_request("short", 16, hasher, seed=2))
    assert scheduler.schedule().num_scheduled_tokens == expected


def test_running_chunks_and_waiting_window_go_shortest_first(model_config):
    scheduler, hasher = _scheduler(model_config, max_num_batched_tokens=64)
    for name, n, seed in (("w48", 48, 1), ("w16", 16, 2), ("w32", 32, 3)):
        scheduler.add_request(_request(name, n, hasher, seed=seed))
    paw_step = scheduler._paw_step
    assert scheduler.schedule().num_scheduled_tokens == {
        "w16": 16,
        "w32": 32,
        "w48": 16,
    }
    # The external-warmth refresh clock still advances once per step.
    assert scheduler._paw_step == paw_step + 1


def test_waiting_prompt_longer_than_the_chunk_gets_no_reserve(model_config):
    scheduler, hasher = _scheduler(model_config, max_num_batched_tokens=32)
    scheduler.add_request(_request("long", 64, hasher, seed=1))
    scheduler.schedule()
    scheduler.add_request(_request("longer", 160, hasher, seed=2))
    assert scheduler.schedule().num_scheduled_tokens == {"long": 32}


def test_starved_chunk_is_no_longer_cut(model_config):
    scheduler, hasher = _scheduler(model_config, max_num_batched_tokens=32)
    scheduler.max_starved_steps = 1
    scheduler.add_request(_request("long", 160, hasher, seed=1))
    scheduler.schedule()
    scheduler.add_request(_request("short", 32, hasher, seed=2))
    # The short prompt takes the whole budget; the chunk gets no block.
    assert scheduler.schedule().num_scheduled_tokens == {"short": 32}
    scheduler.add_request(_request("short2", 32, hasher, seed=3))
    # Starved for max_starved_steps: the chunk is not cut for short2.
    assert scheduler.schedule().num_scheduled_tokens == {"short": 1, "long": 31}


def _decoding_and_long(scheduler, hasher):
    """One request decoding and a long prompt with its first chunk run."""
    scheduler.add_request(_request("dec", 16, hasher, seed=1))
    scheduler.schedule()
    scheduler.add_request(_request("long", 320, hasher, seed=2))
    out = scheduler.schedule()
    assert "long" in out.num_scheduled_tokens and "dec" in out.num_scheduled_tokens
    return out


def test_long_chunk_is_followed_by_decode_only_steps(model_config):
    scheduler, hasher = _scheduler(model_config, max_num_batched_tokens=64)
    scheduler.spacing_frac = 1.0
    scheduler.long_prompt_tokens = 128
    scheduler._step_s = {"long": 0.1, "decode": 0.04}
    _decoding_and_long(scheduler, hasher)
    assert scheduler._debt_s == pytest.approx(0.1)
    # 0.1 s owed at 0.04 s per decode-only step: three of them, then a chunk.
    for _ in range(3):
        assert set(scheduler.schedule().num_scheduled_tokens) == {"dec"}
    assert "long" in scheduler.schedule().num_scheduled_tokens


def test_no_spacing_after_a_short_prompt_or_without_decodes(model_config):
    scheduler, hasher = _scheduler(model_config, max_num_batched_tokens=64)
    scheduler.spacing_frac = 1.0
    scheduler.long_prompt_tokens = 1000
    scheduler._step_s = {"long": 0.1, "decode": 0.04}
    _decoding_and_long(scheduler, hasher)
    assert scheduler._debt_s == 0
    assert "long" in scheduler.schedule().num_scheduled_tokens

    alone, hasher = _scheduler(model_config, max_num_batched_tokens=64)
    alone.spacing_frac = 1.0
    alone.long_prompt_tokens = 128
    alone._step_s = {"long": 0.1, "decode": 0.04}
    alone.add_request(_request("long", 320, hasher, seed=2))
    alone.schedule()
    assert alone._debt_s == 0
    assert "long" in alone.schedule().num_scheduled_tokens


def test_step_time_is_timed_only_back_to_back(model_config, monkeypatch):
    scheduler, _ = _scheduler(model_config)
    monkeypatch.setattr(AsyncScheduler, "update_from_output", lambda *args: {})
    clock = iter([10.0, 10.5, 10.6, 50.0])
    monkeypatch.setattr(pss.time, "monotonic", lambda: next(clock))
    a, b, c = (SimpleNamespace() for _ in range(3))
    scheduler._pending_kinds = {id(a): "long", id(b): "decode"}
    scheduler.update_from_output(a, None)  # t=10.0, b still queued
    scheduler._pending_kinds[id(c)] = "decode"
    scheduler.update_from_output(b, None)  # t=10.5: b took 0.5 s
    assert scheduler._step_s == {"decode": 0.5}
    scheduler.update_from_output(c, None)  # t=10.6: c took 0.1 s; nothing queued
    assert scheduler._step_s["decode"] == pytest.approx(0.5 + 0.2 * (0.1 - 0.5))
    d = SimpleNamespace()
    scheduler._pending_kinds[id(d)] = "decode"
    scheduler.update_from_output(d, None)  # t=50: after idle, not timed
    assert scheduler._step_s["decode"] == pytest.approx(0.42)
