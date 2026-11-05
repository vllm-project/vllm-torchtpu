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
"""Platform gates for the short-prompt-first scheduler."""

from types import SimpleNamespace

from vllm_torchtpu.core.short_prefill_first_scheduler import SCHEDULER_CLS
from vllm_torchtpu.platforms import tpu_platform


def _configure(async_scheduling, scheduler_cls=None):
    config = SimpleNamespace(
        async_scheduling=async_scheduling, scheduler_cls=scheduler_cls
    )
    tpu_platform._configure_short_prefill_first(config)
    return config.scheduler_cls


def test_async_scheduling_gets_the_short_prefill_first_scheduler():
    """The hook must install the scheduler when no other one is configured."""
    assert _configure(True) == SCHEDULER_CLS


def test_sync_scheduling_keeps_the_default_scheduler():
    """An AsyncScheduler subclass must not run under the sync engine step,
    which leaves its draft-token placeholders to the drafter's update."""
    assert _configure(False) is None


def test_a_configured_scheduler_is_kept():
    """The hook must not replace a scheduler the user configured."""
    assert _configure(True, "x.Y") == "x.Y"
