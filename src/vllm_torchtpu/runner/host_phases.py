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

import time

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_EVERY = envs.VLLM_TPU_HOST_PHASES
_last: int | None = None
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
    global _last
    _last = time.perf_counter_ns()


def mark(name: str) -> None:
    global _last
    if not _EVERY:
        return
    now = time.perf_counter_ns()
    if _last is not None:
        _samples.setdefault(_kind, {}).setdefault(name, []).append(now - _last)
        _cur.append((name, now - _last))
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
