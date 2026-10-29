# SPDX-License-Identifier: Apache-2.0
"""Log the scheduler's state when it keeps scheduling zero tokens while
requests are running (on with VLLM_TPU_STALL_WATCHDOG_S > 0).

The c32 freezes: every core idle for tens of seconds while the engine keeps
sending empty steps to the workers. Under async speculative scheduling a
running request schedules nothing until `update_from_output` releases the
output placeholders of its step in flight; if an output never releases them,
the engine spins on empty batches. When such a streak lasts 2 s this logs, per
running request, the counters `schedule()` derives its token count from and
whether the latest `update_from_output` carried the request (then every 10 s
while the streak lasts).
"""

import functools
import os
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_FIRST_LOG_S = 2.0
_REPEAT_LOG_S = 10.0
_MAX_REQS = 16


def _enabled() -> bool:
    try:
        return float(os.environ.get("VLLM_TPU_STALL_WATCHDOG_S", "0") or 0) > 0
    except ValueError:
        return False


def _request_row(request, last_ids) -> str:
    def get(name):
        try:
            return getattr(request, name)
        except Exception:
            return "?"

    return (
        f"{str(request.request_id)[-12:]}: computed={get('num_computed_tokens')} "
        f"placeholders={get('num_output_placeholders')} "
        f"with_spec={get('num_tokens_with_spec')} "
        f"prompt={get('num_prompt_tokens')} max_tokens={get('max_tokens')} "
        f"spec={len(get('spec_token_ids') or ())} status={get('status')} "
        f"in_last_update={request.request_id in last_ids}"
    )


def patch_zero_schedule_diag() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if not _enabled() or getattr(Scheduler.schedule, "_tpu_zero_diag", False):
        return
    original_schedule = Scheduler.schedule
    original_update = Scheduler.update_from_output

    @functools.wraps(original_update)
    def update_from_output(self, scheduler_output, model_runner_output):
        ids = getattr(model_runner_output, "req_ids", None) or ()
        self._tpu_zero_last_update = (time.monotonic(), frozenset(ids))
        return original_update(self, scheduler_output, model_runner_output)

    @functools.wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        out = original_schedule(self, *args, **kwargs)
        if out.total_num_scheduled_tokens or not self.running:
            self._tpu_zero_streak = None
            return out
        now = time.monotonic()
        streak = self.__dict__.get("_tpu_zero_streak")
        if streak is None:
            self._tpu_zero_streak = streak = [now, 0, now + _FIRST_LOG_S]
        streak[1] += 1
        if now >= streak[2]:
            streak[2] = now + _REPEAT_LOG_S
            at, ids = self.__dict__.get("_tpu_zero_last_update", (None, frozenset()))
            rows = [_request_row(r, ids) for r in list(self.running)[:_MAX_REQS]]
            logger.warning(
                "ZERO-SCHEDULE streak: %.1f s, %d empty schedules, %d running, "
                "%d waiting; last update_from_output %s ago with %d req ids\n  %s",
                now - streak[0],
                streak[1],
                len(self.running),
                len(self.waiting),
                "never" if at is None else f"{now - at:.1f} s",
                len(ids),
                "\n  ".join(rows),
            )
        return out

    schedule._tpu_zero_diag = True
    Scheduler.schedule = schedule
    Scheduler.update_from_output = update_from_output
    logger.info("Zero-schedule streak diagnostic on")
