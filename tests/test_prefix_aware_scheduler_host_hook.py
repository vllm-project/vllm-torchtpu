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
"""PrefixAwareScheduler.schedule() runs the host-process hook
(host_trims.on_scheduler_step) with the engine's config before anything else,
on a kv_consumer (decode-leg) engine and on any other, and skips it on a
scheduler built without a config."""

import time
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.runner import host_trims


def _scheduler(monkeypatch, events, kv_consumer):
    """A stand-in with what schedule() reads on each KV-role branch; vLLM's
    own schedule() and the hook record the order they run in."""

    def upstream_schedule(self, *args, **kwargs):
        events.append("schedule")
        return "out"

    monkeypatch.setattr(AsyncScheduler, "schedule", upstream_schedule, raising=False)
    monkeypatch.setattr(
        host_trims, "on_scheduler_step", lambda cfg: events.append(("hook", cfg))
    )
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._is_kv_consumer = kv_consumer
    s.running = []
    s.waiting = []
    s.kv_holding_waiting = []
    # The tpu_scheduler options schedule() reads: KV prefetch off (no count,
    # no cap), the warmest-first promotion on (its default).
    s._kv_prefetch_n = 0
    s._kv_prefetch_cap = None
    s._promote_warmest_switch = True
    # Off a decode leg: the warmest-first promotion.
    s._promote_warmest = lambda queue: None
    # On a decode leg: the queue selection and the step telemetry, with the
    # periodic log not due.
    s._select_waiting_queue_for_scheduling = lambda: None
    s._waiting_loop_stop_reason = lambda out, waiting_at_start: "idle"
    s._record_step_telemetry = lambda out, waiting_at_start, stop: None
    s._last_sched_log_t = time.monotonic() + 1e9
    return s


@pytest.mark.parametrize("kv_consumer", [True, False])
def test_schedule_runs_the_host_hook_first_with_the_engine_config(
    monkeypatch, kv_consumer
):
    events = []
    s = _scheduler(monkeypatch, events, kv_consumer)
    s.vllm_config = SimpleNamespace()
    assert s.schedule() == "out"
    assert events == [("hook", s.vllm_config), "schedule"]


@pytest.mark.parametrize("kv_consumer", [True, False])
def test_schedule_skips_the_host_hook_without_an_engine_config(
    monkeypatch, kv_consumer
):
    events = []
    s = _scheduler(monkeypatch, events, kv_consumer)
    assert s.schedule() == "out"
    assert events == ["schedule"]
