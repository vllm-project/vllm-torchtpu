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
"""``tpu_scheduler.promote_warmest``: warmest-first promotion of the waiting
queues can be turned off (FCFS admission). Default on.
"""

import inspect
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_torchtpu.core import scheduler_options
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

# ---------------- tpu_scheduler.promote_warmest ----------------


@pytest.mark.parametrize("val,expect", [(None, True), (True, True), (False, False)])
def test_promote_switch(val, expect):
    section = {} if val is None else {"promote_warmest": val}
    cfg = SimpleNamespace(additional_config={"tpu_scheduler": section})
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._promote_warmest_switch = scheduler_options.get(cfg).promote_warmest
    assert s._promote_warmest_on() is expect
    assert PrefixAwareScheduler.__new__(PrefixAwareScheduler)._promote_warmest_on()


def test_pas_schedule_guards_the_promotion():
    src = inspect.getsource(PrefixAwareScheduler.schedule)
    assert (
        "if self._promote_warmest_on():\n"
        "                self._select_waiting_queue_for_scheduling()" in src
    )


@pytest.mark.parametrize("on", [True, False])
def test_schedule_promotes_only_when_on(monkeypatch, on):
    calls = []
    monkeypatch.setattr(AsyncScheduler, "schedule", lambda self, *a, **k: "out")
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.running = []
    s._promote_warmest_switch = on
    s._select_waiting_queue_for_scheduling = lambda: calls.append(1)
    assert s.schedule() == "out"
    assert calls == ([1] if on else [])


def test_init_reads_the_option(monkeypatch):
    from unittest.mock import MagicMock

    from vllm_torchtpu.core import prefix_change_log

    def fake_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            kv_transfer_config=None,
            additional_config={"tpu_scheduler": {"promote_warmest": False}},
        )
        self.connector = None
        self.kv_cache_manager = MagicMock()

    monkeypatch.setattr(AsyncScheduler, "__init__", fake_init)
    monkeypatch.setattr(prefix_change_log, "install_epoch", lambda: None)
    monkeypatch.setattr(PrefixAwareScheduler, "_enable_memo_xstep", lambda self: None)
    assert PrefixAwareScheduler()._promote_warmest_on() is False
