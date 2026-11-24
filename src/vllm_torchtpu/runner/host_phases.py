# SPDX-License-Identifier: Apache-2.0
"""Per-step host time of the runner's phases (VLLM_TPU_HOST_PHASES=N).

``mark(name)`` charges the host time since the previous mark to ``name``;
``begin_step()`` charges the time since the previous step's last mark to
``idle`` (engine core, IPC and scheduling between runner calls). Every N
steps each rank logs p50 / p90 / mean microseconds per phase, so the ranks'
host budgets can be compared without a profiler. A mark costs about a
microsecond; with N=0 (the default) every call returns at once.

Steps are bucketed by kind (``begin_step(kind)``): ``admit`` (the step adds a
new request), ``pf`` (a prefill chunk, no admission) and ``dec`` (decode
only), so the mixed-step host bubble is not averaged away by the decode
steps. Rank 0 also logs one ``HOSTADMIT`` line per admit step with that
step's phases and, when the scheduler stamped it, the engine-side schedule
cost and the schedule-end to runner-start lag (see
``core/prefix_aware_scheduler.py``).
"""

import os
import threading
import time

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_EVERY = envs.VLLM_TPU_HOST_PHASES
# Also charge each phase's on-CPU time of the step thread to x_cpu_<phase>.
_CPU = _EVERY and envs.VLLM_TPU_HOST_PHASES_CPU
_last: int | None = None
_last_cpu = 0
_samples: dict[str, dict[str, list[int]]] = {}
_kind_steps: dict[str, int] = {}
_steps = 0
_rank = -1
_kind = "all"
_cur: list[tuple[str, int]] = []
_cur_info = ""


def set_rank(rank: int) -> None:
    global _rank
    _rank = rank
    if _EVERY:
        _install_gc_timer()


_gc_start = 0


def _gc_callback(phase: str, info: dict) -> None:
    """Charge each cyclic-GC pass to ``x_gc<generation>`` of the current step."""
    global _gc_start
    if phase == "start":
        _gc_start = time.perf_counter_ns()
    elif _gc_start:
        note(f"x_gc{info.get('generation', '?')}", (time.perf_counter_ns() - _gc_start) / 1e3)
        _gc_start = 0


def _install_gc_timer() -> None:
    import gc

    if _gc_callback not in gc.callbacks:
        gc.callbacks.append(_gc_callback)


def enabled() -> bool:
    return bool(_EVERY)


def begin_step(kind: str = "all", info: str = "") -> None:
    """Start a step of ``kind``; the host time since the previous step's last
    mark (engine core, IPC, scheduling) is charged to this step as ``idle``."""
    global _steps, _kind, _cur, _cur_info
    if not _EVERY:
        return
    prev_kind, prev_cur, prev_info = _kind, _cur, _cur_info
    _kind, _cur, _cur_info = kind, [], info
    if _last is not None:
        mark("idle")
    else:
        _set_last()
    if prev_kind == "admit" and _rank <= 0 and prev_cur:
        _log_admit(prev_cur, prev_info)
    _kind_steps[kind] = _kind_steps.get(kind, 0) + 1
    _steps += 1
    if _steps % _EVERY == 0:
        _report()


def note(name: str, value_us: float) -> None:
    """Record an externally measured value (us) for this step's kind."""
    if not _EVERY:
        return
    ns = int(value_us * 1e3)
    _samples.setdefault(_kind, {}).setdefault(name, []).append(ns)
    _cur.append((name, ns))


def timed(fn, label: str):
    """Wrap ``fn`` so each call's host time is recorded as ``label``."""

    def wrapper(*args, **kwargs):
        t = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            note(label, (time.perf_counter_ns() - t) / 1e3)

    return wrapper


def _log_admit(phases: list[tuple[str, int]], info: str) -> None:
    total = sum(ns for name, ns in phases if not name.startswith("x_"))
    logger.info(
        "HOSTADMIT rank=%d %s total_us=%.0f %s",
        _rank,
        info,
        total / 1e3,
        " ".join(f"{name}={ns / 1e3:.0f}" for name, ns in phases),
    )


def _set_last() -> None:
    global _last, _last_cpu
    _last = time.perf_counter_ns()
    if _CPU:
        _last_cpu = time.thread_time_ns()


def mark(name: str) -> None:
    global _last, _last_cpu
    if not _EVERY:
        return
    now = time.perf_counter_ns()
    if _last is not None:
        per_name = _samples.setdefault(_kind, {})
        per_name.setdefault(name, []).append(now - _last)
        _cur.append((name, now - _last))
        if _CPU:
            cpu = time.thread_time_ns()
            per_name.setdefault("x_cpu_" + name, []).append(cpu - _last_cpu)
            _last_cpu = cpu
    _last = now


def _report() -> None:
    for kind, per_name in _samples.items():
        n_steps = max(_kind_steps.get(kind, 0), 1)
        parts = []
        total = 0.0
        for name, values in per_name.items():
            v = np.asarray(values, dtype=np.float64) / 1e3
            mean = float(v.sum()) / n_steps
            if not name.startswith("x_"):
                total += mean
            parts.append(
                f"{name}={np.percentile(v, 50):.0f}/{np.percentile(v, 90):.0f}"
                f"/{mean:.0f}"
            )
        logger.info(
            "HOSTPHASES rank=%d steps=%d kind=%s kind_steps=%d per-step host us "
            "(p50/p90/mean-per-step) total_mean=%.0f %s",
            _rank,
            _steps,
            kind,
            n_steps,
            total,
            " ".join(parts),
        )
    _samples.clear()
    _kind_steps.clear()
    log_thread_usage(f"rank={_rank}")


# tid -> (name, run ns, runqueue-wait ns, timeslices) at the previous report.
_thread_prev: dict[int, tuple[str, int, int, int]] = {}
_thread_prev_wall = 0


def _read_threads() -> dict[int, tuple[str, int, int, int]]:
    out = {}
    for tid in os.listdir("/proc/self/task"):
        base = f"/proc/self/task/{tid}/"
        try:
            with open(base + "comm") as f:
                name = f.read().strip()
            with open(base + "schedstat") as f:
                run_ns, wait_ns, slices = (int(x) for x in f.read().split())
        except (FileNotFoundError, ProcessLookupError):
            continue  # the thread exited between listdir and open
        out[int(tid)] = (name, run_ns, wait_ns, slices)
    return out


def log_thread_usage(tag: str) -> None:
    """Log each thread's on-CPU and runqueue-wait time since the previous call
    (``/proc/self/task/*/schedstat``), so a step thread that waits for a CPU,
    or another thread of the process that keeps one busy, shows next to the
    phase times. ``*`` marks the calling thread."""
    global _thread_prev, _thread_prev_wall
    now = time.perf_counter_ns()
    cur = _read_threads()
    wall = now - _thread_prev_wall
    if _thread_prev_wall:
        me = threading.get_native_id()
        rows = []
        for tid, (name, run, wait, slices) in cur.items():
            _, run0, wait0, slices0 = _thread_prev.get(tid, (name, 0, 0, 0))
            d_run, d_wait = run - run0, wait - wait0
            if tid == me or d_run + d_wait > wall // 200:
                rows.append((d_run, tid, name, d_wait, slices - slices0))
        rows.sort(reverse=True)
        logger.info(
            "HOSTTHREADS %s wall_ms=%.0f threads=%d %s",
            tag,
            wall / 1e6,
            len(cur),
            " ".join(
                f"{'*' if tid == me else ''}{name.replace(' ', '_')}/{tid}="
                f"{d_run / 1e6:.0f}/{d_wait / 1e6:.0f}/{n}"
                for d_run, tid, name, d_wait, n in rows
            ),
        )
    _thread_prev, _thread_prev_wall = cur, now
