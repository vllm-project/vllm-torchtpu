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
(host_trims.on_scheduler_step) with the engine's config, and skips it on a
scheduler built without one."""

from types import SimpleNamespace

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.runner import host_trims


def _scheduler(monkeypatch, calls):
    monkeypatch.setattr(
        AsyncScheduler, "schedule", lambda self, *a, **k: "out", raising=False
    )
    monkeypatch.setattr(host_trims, "on_scheduler_step", calls.append)
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.running = []
    s.connector = SimpleNamespace()
    return s


def test_schedule_runs_the_host_hook_once_with_the_engine_config(monkeypatch):
    calls = []
    s = _scheduler(monkeypatch, calls)
    s.vllm_config = SimpleNamespace()
    assert s.schedule() == "out"
    assert len(calls) == 1
    assert calls[0] is s.vllm_config


def test_schedule_skips_the_host_hook_without_an_engine_config(monkeypatch):
    calls = []
    s = _scheduler(monkeypatch, calls)
    assert s.schedule() == "out"
    assert calls == []
