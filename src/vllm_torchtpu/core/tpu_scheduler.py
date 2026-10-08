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
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

_PROCESS_CROSS_DP_HAS_ACTIVE_DECODE: bool = False


def set_process_cross_dp_active_decode(active: bool) -> None:
    """Update the process-level cross-DP active decode indicator."""
    global _PROCESS_CROSS_DP_HAS_ACTIVE_DECODE
    _PROCESS_CROSS_DP_HAS_ACTIVE_DECODE = bool(active)


def get_process_cross_dp_active_decode() -> bool:
    """Return the process-level cross-DP active decode indicator."""
    return bool(_PROCESS_CROSS_DP_HAS_ACTIVE_DECODE)


class TpuDpScheduler(AsyncScheduler):
    """Scheduler that aligns prefill admission across DP ranks.

    vLLM passes a DP-synchronized scheduling signal through ``throttle_prefills``.
    This scheduler uses that signal only to decide when to admit new prefill
    requests until all pending prefills are drained.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drain_prefill = False
        logger.info("TpuDpScheduler active")

    def _schedule_paused_new(self) -> SchedulerOutput:
        """Run one step while keeping newly prefills buffered."""
        prev_pause_state = self._pause_state
        if prev_pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            return super().schedule(False)
        finally:
            self._pause_state = prev_pause_state

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.prefill_capacity_bound = True
        if not throttle_prefills:
            # Open window for draining accumulated prefills.
            self._drain_prefill = True
        if self._drain_prefill and not (self.waiting or self.skipped_waiting):
            # In drain mode until all accumulated prefills are scheduled.
            self._drain_prefill = False
        if self._drain_prefill:
            return super().schedule(False)
        return self._schedule_paused_new()
