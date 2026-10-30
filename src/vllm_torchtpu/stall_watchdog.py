# SPDX-License-Identifier: Apache-2.0
"""Dump every process's Python stacks when serving stalls (VLLM_TPU_STALL_WATCHDOG_S=N).

Rare multi-minute full-server freezes (no program issued on any rank, the API
server silent, requests still running, then a clean resume) leave nothing in
the logs. With N > 0:

- every vLLM process (API server, engine core, workers; via the general
  plugin) registers ``faulthandler`` on SIGUSR2, so the signal prints all its
  threads' Python stacks to stderr without stopping it;
- each worker watches its own steps: when requests are still live in its
  batch but no step with scheduled tokens has started for N seconds (empty
  schedules do not count), it dumps its stacks once per
  stall, and rank 0 also signals its parent (engine core) and grandparent
  (API server) so their stacks land in the same log;
- the API server watches its asyncio event loop: a task re-arms
  ``faulthandler.dump_traceback_later`` every second, so a loop blocked for
  VLLM_TPU_LOOP_LAG_WATCHDOG_S (default 30) dumps its stacks once per stall,
  even when the blocking code holds the GIL. The block length is logged when
  the loop resumes. The engine can be idle during such a stall (outputs pile
  up unread), so the step watch above cannot see it.
"""

import asyncio
import faulthandler
import multiprocessing
import os
import signal
import sys
import threading
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_last_step = 0.0
_live_reqs = 0
# Set once warmup is over: warmup's real-path requests legitimately pause on
# compiles for tens of seconds.
_armed = False


def _seconds() -> float:
    try:
        return float(os.environ.get("VLLM_TPU_STALL_WATCHDOG_S", "0") or 0)
    except ValueError:
        return 0.0


_installed = False


def install_signal_dump() -> None:
    """Register the SIGUSR2 stack dump in this process (any role), once."""
    global _installed
    if _seconds() <= 0 or _installed:
        return
    _installed = True
    try:
        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
    except (AttributeError, ValueError, RuntimeError):
        return
    # Plugins load before vLLM configures logging; print so the line shows.
    print(f"Stall watchdog: SIGUSR2 dumps the stacks of pid {os.getpid()}",
          file=sys.stderr, flush=True)
    # Engine core and workers are multiprocessing children; the API server
    # is the process `vllm serve` started.
    if _loop_lag_seconds() > 0 and multiprocessing.parent_process() is None:
        threading.Thread(
            target=_hook_api_server_loop, name="loop-lag-hook", daemon=True
        ).start()


def arm() -> None:
    """Start watching (the worker calls this when warmup completes)."""
    global _armed
    _armed = True


def note_step(live_reqs: int) -> None:
    """Called by the runner as each step starts."""
    global _last_step, _live_reqs
    _last_step = time.monotonic()
    _live_reqs = live_reqs


def note_live(live_reqs: int) -> None:
    """A step that scheduled no tokens: keeps the live-request count current
    without counting as progress."""
    global _live_reqs
    _live_reqs = live_reqs


def _parent(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def _catches_sigusr2(pid: int) -> bool:
    """Whether ``pid`` installed a SIGUSR2 handler. The signal's default action
    terminates, so a process without the dump handler is never signalled."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("SigCgt:"):
                    return bool(int(line.split()[1], 16) & (1 << (signal.SIGUSR2 - 1)))
    except (OSError, ValueError):
        pass
    return False


def _watch(limit: float, rank: int) -> None:
    dumped = False
    while True:
        time.sleep(min(5.0, limit / 4))
        if not _armed or not _last_step:
            continue
        idle = time.monotonic() - _last_step
        if idle < limit or _live_reqs == 0:
            dumped = False
            continue
        if dumped:
            continue
        dumped = True
        logger.warning(
            "STALL rank=%d: no step for %.0f s with %d live requests; dumping stacks",
            rank,
            idle,
            _live_reqs,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        if rank == 0:
            pid = os.getpid()
            for _ in range(2):
                pid = _parent(pid)
                if pid <= 1 or not _catches_sigusr2(pid):
                    break
                try:
                    os.kill(pid, signal.SIGUSR2)
                    logger.warning("STALL: sent SIGUSR2 to pid %d", pid)
                except OSError:
                    break


def start(rank: int) -> None:
    """Start the worker-side stall watch (no-op unless enabled)."""
    limit = _seconds()
    if limit <= 0:
        return
    install_signal_dump()
    threading.Thread(
        target=_watch, args=(limit, rank), name="stall-watchdog", daemon=True
    ).start()
    logger.info(
        "Stall watchdog on: rank %d dumps stacks after %.0f s without a step", rank, limit
    )


def _loop_lag_seconds() -> float:
    try:
        return float(os.environ.get("VLLM_TPU_LOOP_LAG_WATCHDOG_S", "30") or 0)
    except ValueError:
        return 0.0


_loop_watch_started = False


def _hook_api_server_loop() -> None:
    """Wrap AsyncLLM._run_output_handler once vLLM has imported it (importing
    it from the plugin itself would be circular). The handler is (re)entered
    on every generate() call, on the server's event loop, so the first request
    after the wrap starts the loop watch."""
    for _ in range(3600):
        mod = sys.modules.get("vllm.v1.engine.async_llm")
        cls = getattr(mod, "AsyncLLM", None)
        if cls is not None:
            break
        time.sleep(1.0)
    else:
        return
    original = cls._run_output_handler

    def _run_output_handler(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not _loop_watch_started:
            try:
                start_loop_watch(asyncio.get_running_loop(), _loop_lag_seconds())
            except RuntimeError:  # not on a running loop yet
                pass
        return result

    cls._run_output_handler = _run_output_handler


def start_loop_watch(loop: asyncio.AbstractEventLoop, limit: float) -> None:
    """Watch ``loop`` (call on it): dump all stacks if it is blocked for
    ``limit`` seconds, log the block length when it resumes."""
    global _loop_watch_started
    if _loop_watch_started or limit <= 0:
        return
    _loop_watch_started = True
    loop.create_task(_loop_watch(limit))
    logger.info(
        "Loop-lag watchdog on: pid %d dumps stacks when its event loop blocks for %.0f s",
        os.getpid(),
        limit,
    )


async def _loop_watch(limit: float) -> None:
    # faulthandler's timer thread runs without the GIL, so the dump fires
    # even while the loop is stuck in C code that holds it.
    period = min(1.0, limit / 4)
    try:
        while True:
            faulthandler.dump_traceback_later(limit, repeat=False, file=sys.stderr)
            start = time.monotonic()
            await asyncio.sleep(period)
            lag = time.monotonic() - start - period
            if lag >= limit:
                logger.warning(
                    "LOOP STALL pid=%d: event loop was blocked for %.1f s (stacks dumped above)",
                    os.getpid(),
                    lag,
                )
    finally:
        # Loop shutdown: a slow teardown is not a stall.
        faulthandler.cancel_dump_traceback_later()
