from unittest.mock import patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState

from vllm_torchtpu.core.tpu_scheduler import TpuDpScheduler


def _make_scheduler(waiting=(), drain_prefill=False):
    scheduler = TpuDpScheduler.__new__(TpuDpScheduler)
    scheduler.waiting = list(waiting)
    scheduler.skipped_waiting = []
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler._drain_prefill = drain_prefill
    scheduler.prefill_capacity_bound = False
    return scheduler


def test_unthrottled_step_opens_drain_window():
    scheduler = _make_scheduler(waiting=[object()])
    expected = object()

    with patch.object(AsyncScheduler, "schedule",
                      return_value=expected) as schedule:
        result = scheduler.schedule(throttle_prefills=False)

    assert result is expected
    schedule.assert_called_once_with(False)
    assert scheduler._drain_prefill is True
    # Upstream throttling is neutralized so the cadence is the only clock.
    assert scheduler.prefill_capacity_bound is True


def test_throttled_step_pauses_new_requests():
    scheduler = _make_scheduler(waiting=[object()])
    expected = object()
    seen = {}

    def capture(*args, **kwargs):
        seen["pause_state"] = scheduler._pause_state
        return expected

    with patch.object(AsyncScheduler, "schedule",
                      side_effect=capture) as schedule:
        result = scheduler.schedule(throttle_prefills=True)

    assert result is expected
    schedule.assert_called_once_with(False)
    assert seen["pause_state"] is PauseState.PAUSED_NEW
    assert scheduler._pause_state is PauseState.UNPAUSED


def test_drain_spans_throttled_steps_until_queue_empties():
    scheduler = _make_scheduler(waiting=[object()], drain_prefill=True)
    seen = {}

    def capture(*args, **kwargs):
        seen["pause_state"] = scheduler._pause_state
        return object()

    # A throttled step keeps draining while requests are still waiting.
    with patch.object(AsyncScheduler, "schedule", side_effect=capture):
        scheduler.schedule(throttle_prefills=True)

    assert seen["pause_state"] is PauseState.UNPAUSED
    assert scheduler._drain_prefill is True

    # Once the queue drains, the window closes and buffering resumes.
    scheduler.waiting = []
    with patch.object(AsyncScheduler, "schedule", side_effect=capture):
        scheduler.schedule(throttle_prefills=True)

    assert seen["pause_state"] is PauseState.PAUSED_NEW
    assert scheduler._drain_prefill is False


def test_pause_state_restored_when_schedule_raises():
    scheduler = _make_scheduler(waiting=[object()])

    with patch.object(AsyncScheduler, "schedule", side_effect=RuntimeError):
        try:
            scheduler.schedule(throttle_prefills=True)
        except RuntimeError:
            pass

    assert scheduler._pause_state is PauseState.UNPAUSED
