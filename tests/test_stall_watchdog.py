# SPDX-License-Identifier: Apache-2.0
import os
import threading
import time

from vllm_torchtpu import stall_watchdog


def test_signal_handler_is_visible_in_sigcgt(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_STALL_WATCHDOG_S", "60")
    stall_watchdog.install_signal_dump()
    assert stall_watchdog._catches_sigusr2(os.getpid())
    assert not stall_watchdog._catches_sigusr2(1) or os.getpid() == 1


def test_dumps_once_per_stall_only_with_live_requests(monkeypatch):
    dumps = []
    monkeypatch.setattr(
        stall_watchdog.faulthandler, "dump_traceback", lambda **kw: dumps.append(1)
    )
    monkeypatch.setattr(stall_watchdog, "_armed", True)
    stall_watchdog.note_step(0)
    stall_watchdog._last_step = time.monotonic() - 10
    threading.Thread(target=stall_watchdog._watch, args=(0.2, 3), daemon=True).start()
    time.sleep(0.3)
    assert dumps == []  # idle server: no live requests
    stall_watchdog._live_reqs = 2
    time.sleep(0.3)
    assert dumps == [1]  # one dump per stall, not one per poll
    stall_watchdog.note_step(2)  # steps resume, then stall again
    time.sleep(0.4)
    assert dumps == [1, 1]


def test_empty_schedules_do_not_count_as_progress(monkeypatch):
    """Zero-token steps keep arriving during a freeze; note_live updates the
    live count without resetting the stall clock, so the dump still fires."""
    dumps = []
    monkeypatch.setattr(
        stall_watchdog.faulthandler, "dump_traceback", lambda **kw: dumps.append(1)
    )
    monkeypatch.setattr(stall_watchdog, "_armed", True)
    stall_watchdog.note_step(4)
    stall_watchdog._last_step = time.monotonic() - 10
    stop = threading.Event()

    def empty_steps():
        while not stop.is_set():
            stall_watchdog.note_live(4)
            time.sleep(0.01)

    threading.Thread(target=empty_steps, daemon=True).start()
    threading.Thread(target=stall_watchdog._watch, args=(0.2, 3), daemon=True).start()
    time.sleep(0.4)
    stop.set()
    assert dumps == [1]
