# SPDX-License-Identifier: Apache-2.0
"""Per-step host time of the runner's phases (VLLM_TPU_HOST_PHASES=N).

``mark(name)`` charges the host time since the previous mark to ``name``;
``begin_step()`` charges the time since the previous step's last mark to
``idle`` (engine core, IPC and scheduling between runner calls). Every N
steps each rank logs p50 / p90 / mean microseconds per phase, so the ranks'
host budgets can be compared without a profiler. A mark costs about a
microsecond; with N=0 (the default) every call returns at once.
"""

import time

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_EVERY = envs.VLLM_TPU_HOST_PHASES
_last: int | None = None
_samples: dict[str, list[int]] = {}
_steps = 0
_rank = -1


def set_rank(rank: int) -> None:
    global _rank
    _rank = rank


def begin_step() -> None:
    global _steps
    if not _EVERY:
        return
    if _last is not None:
        mark("idle")
    else:
        _set_last()
    _steps += 1
    if _steps % _EVERY == 0:
        _report()


def _set_last() -> None:
    global _last
    _last = time.perf_counter_ns()


def mark(name: str) -> None:
    global _last
    if not _EVERY:
        return
    now = time.perf_counter_ns()
    if _last is not None:
        _samples.setdefault(name, []).append(now - _last)
    _last = now


def _report() -> None:
    parts = []
    total = 0.0
    for name, values in _samples.items():
        v = np.asarray(values, dtype=np.float64) / 1e3
        mean = float(v.sum()) / _EVERY
        total += mean
        parts.append(
            f"{name}={np.percentile(v, 50):.0f}/{np.percentile(v, 90):.0f}/{mean:.0f}"
        )
    logger.info(
        "HOSTPHASES rank=%d steps=%d per-step host us (p50/p90/mean-per-step) "
        "total_mean=%.0f %s",
        _rank,
        _steps,
        total,
        " ".join(parts),
    )
    _samples.clear()
