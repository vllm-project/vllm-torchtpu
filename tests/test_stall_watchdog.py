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


def test_loop_watch_dumps_when_the_loop_blocks(monkeypatch, tmp_path):
    import asyncio
    import sys

    out = open(tmp_path / "dump.txt", "w+")
    monkeypatch.setattr(sys, "stderr", out)
    monkeypatch.setattr(stall_watchdog, "_loop_watch_started", False)
    warnings = []
    monkeypatch.setattr(
        stall_watchdog.logger, "warning", lambda *a: warnings.append(a)
    )

    async def main():
        stall_watchdog.start_loop_watch(asyncio.get_running_loop(), 0.3)
        await asyncio.sleep(0.5)  # healthy loop: no dump
        assert out.tell() == 0
        time.sleep(0.8)  # block the loop past the limit
        await asyncio.sleep(1.2)  # the watch resumes and logs the block
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    asyncio.run(main())
    out.seek(0)
    dump = out.read()
    assert "Timeout" in dump and "test_loop_watch_dumps_when_the_loop_blocks" in dump
    assert dump.count("Timeout") == 1  # once per stall
    assert len(warnings) == 1 and "LOOP STALL" in warnings[0][0]


def test_loop_watch_hooks_async_llm_output_handler(monkeypatch):
    import asyncio
    import sys
    import types

    started = []
    monkeypatch.setattr(stall_watchdog, "_loop_watch_started", False)
    monkeypatch.setattr(
        stall_watchdog, "start_loop_watch", lambda loop, limit: started.append(limit)
    )
    monkeypatch.setenv("VLLM_TPU_LOOP_LAG_WATCHDOG_S", "45")

    class AsyncLLM:
        def _run_output_handler(self):
            return "ran"

    mod = types.ModuleType("vllm.v1.engine.async_llm")
    mod.AsyncLLM = AsyncLLM
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", mod)
    stall_watchdog._hook_api_server_loop()
    assert AsyncLLM()._run_output_handler() == "ran"  # no running loop: skipped
    assert started == []

    async def main():
        return AsyncLLM()._run_output_handler()

    assert asyncio.run(main()) == "ran"
    assert started == [45.0]
