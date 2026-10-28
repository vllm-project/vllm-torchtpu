# SPDX-License-Identifier: Apache-2.0
"""Dump every process's Python stacks when serving stalls (VLLM_TPU_STALL_WATCHDOG_S=N).

Rare multi-minute full-server freezes (no program issued on any rank, the API
server silent, requests still running, then a clean resume) leave nothing in
the logs. With N > 0:

- every vLLM process (API server, engine core, workers; via the general
  plugin) registers ``faulthandler`` on SIGUSR2, so the signal prints all its
  threads' Python stacks to stderr without stopping it;
- each worker watches its own steps: when requests are still live in its
  batch but no step has started for N seconds, it dumps its stacks once per
  stall, and rank 0 also signals its parent (engine core) and grandparent
  (API server) so their stacks land in the same log.
"""

import faulthandler
import os
import signal
import sys
import threading
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_last_step = 0.0
_live_reqs = 0


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
    logger.info("Stall watchdog: SIGUSR2 dumps the stacks of pid %d", os.getpid())


def note_step(live_reqs: int) -> None:
    """Called by the runner as each step starts."""
    global _last_step, _live_reqs
    _last_step = time.monotonic()
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
        if not _last_step:
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
