# SPDX-License-Identifier: Apache-2.0
"""Per-step host time of the runner's phases (VLLM_TPU_HOST_PHASES=N).

``mark(name)`` charges the host time since the previous mark to ``name``;
``begin_step()`` charges the time since the previous step's last mark to
``idle`` (engine core, IPC and scheduling between runner calls). Every N
steps each rank logs p50 / p90 / mean microseconds per phase, so the ranks'
host budgets can be compared without a profiler. A mark costs about a
microsecond; with N=0 (the default) every call returns at once.

Slow-step dump (q38 DEC-HOST-B; VLLM_TPU_HOST_PHASES_SLOW_MS=T with N > 0):
every step whose host time (the idle before it plus its own phases) is at
least T ms is logged on its own ``HOSTSLOW`` line with its phase breakdown,
the context the runner noted for it (``note``), and sub-phases (``sub`` /
``charge``, names prefixed ``~``; they split a phase, so they are not added
to the total). The aggregate HOSTPHASES lines are unchanged.

Sub-phase census (q38 L4 host sub-mark instrument; VLLM_TPU_HOST_PHASES_SUBS=1
with N > 0): the same ``sub`` / ``charge`` sub-phases are summed per step and,
every N steps, logged on a separate ``HOSTSUBS`` line per rank (p50 / p90 over
the steps that had the sub-phase, mean per step over all N steps, and that step
count). Sub-phases split a phase, so they are never added to HOSTPHASES; the
HOSTPHASES line is byte-identical with the census on or off.

GC census (q38 L6-INSTR, H1 rung 1; with N > 0, unless
VLLM_TPU_HOST_PHASES_GC=0): the worker's first ``begin_step`` appends a
``gc.callbacks`` timer, and every N steps each rank logs the window's cyclic GC
pauses on its own ``GCCENSUS`` line, after HOSTPHASES (and HOSTSUBS):
``gc_us=<sum us>/<collections>/<gen-2 collections>``, the longest pause and
the per-generation split. The pauses belong to whatever ran when the GC
triggered (any thread of the process; the GC holds the GIL), so they are inside
the phases, never added to them. With N = 0 nothing is installed.
"""

import gc
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
# Slow-step dump state: the current step's record (phase -> ns; "~" entries
# are sub-phases), its noted context, and the last sub-phase boundary.
_SLOW_NS = int(envs.VLLM_TPU_HOST_PHASES_SLOW_MS * 1e6) if _EVERY else 0
_cur: dict[str, int] = {}
_ctx: list[str] = []
_sub_last: int | None = None
# Wall time (time.time) at the current record's begin_step, so HOSTSLOW lines
# can be matched against engine-core timestamps.
_cur_wall = 0.0
# Sub-phase census state: the current step's per-sub-phase sums (ns) and, per
# window, one sample per step that had the sub-phase.
_SUBS = bool(envs.VLLM_TPU_HOST_PHASES_SUBS) and bool(_EVERY)
_sub_acc: dict[str, int] = {}
_sub_samples: dict[str, list[int]] = {}
# GC census state (q38 L6-INSTR): the open collection's start, and per window
# [sum ns, collections, gen-2 collections, longest ns] plus per generation
# [ns, collections]. The callback keeps its own clock reference, so a test
# that swaps time.perf_counter_ns for a virtual clock never feeds it.
_GC = bool(_EVERY) and bool(envs.VLLM_TPU_HOST_PHASES_GC)
_gc_clock = time.perf_counter_ns
_gc_hooked = False
_gc_t0: int | None = None
_gc_win = [0, 0, 0, 0]
_gc_gen = [[0, 0], [0, 0], [0, 0]]


def set_rank(rank: int) -> None:
    global _rank
    _rank = rank


def slow_enabled() -> bool:
    return bool(_SLOW_NS)


def note(text: str) -> None:
    """Attach context to the current step's slow-step record."""
    if _SLOW_NS:
        _ctx.append(text)


def sub(name: str) -> None:
    """Charge the time since the last mark or sub to sub-phase ``~name``."""
    global _sub_last
    if not (_SLOW_NS or _SUBS):
        return
    now = time.perf_counter_ns()
    base = _sub_last if _sub_last is not None else _last
    if base is not None:
        key = "~" + name
        if _SLOW_NS:
            _cur[key] = _cur.get(key, 0) + now - base
        if _SUBS:
            _sub_acc[name] = _sub_acc.get(name, 0) + now - base
    _sub_last = now


def charge(name: str, ns: int) -> None:
    """Add ``ns`` to sub-phase ``~name`` without moving any boundary."""
    if _SLOW_NS:
        key = "~" + name
        _cur[key] = _cur.get(key, 0) + ns
    if _SUBS:
        _sub_acc[name] = _sub_acc.get(name, 0) + ns


def begin_step() -> None:
    global _steps, _cur, _ctx, _cur_wall
    if not _EVERY:
        return
    if _SUBS and _sub_acc:
        for name, ns in _sub_acc.items():
            _sub_samples.setdefault(name, []).append(ns)
        _sub_acc.clear()
    if _last is not None:
        if _SLOW_NS:
            # The idle charged below opens the new step's record.
            done, done_ctx, done_wall = _cur, _ctx, _cur_wall
            _cur, _ctx = {}, []
            mark("idle")
            _cur_wall = time.time()
            _log_slow(done, done_ctx, done_wall)
        else:
            mark("idle")
    else:
        _set_last()
        if _GC and not _gc_hooked:
            _gc_install()
    _steps += 1
    if _steps % _EVERY == 0:
        _report()


def _set_last() -> None:
    global _last
    _last = time.perf_counter_ns()


def mark(name: str) -> None:
    global _last, _sub_last
    if not _EVERY:
        return
    now = time.perf_counter_ns()
    if _last is not None:
        _samples.setdefault(name, []).append(now - _last)
        if _SLOW_NS:
            _cur[name] = _cur.get(name, 0) + now - _last
    _last = now
    _sub_last = None


def _log_slow(record: dict[str, int], ctx: list[str], wall: float) -> None:
    total = sum(v for k, v in record.items() if not k.startswith("~"))
    if total < _SLOW_NS:
        return
    logger.info(
        "HOSTSLOW rank=%d step=%d t_begin=%.6f total_us=%d ctx=[%s] %s",
        _rank,
        _steps,
        wall,
        total // 1000,
        " ".join(ctx),
        " ".join(f"{k}={v // 1000}" for k, v in record.items()),
    )


def _phase_stats(samples: dict[str, list[int]]) -> dict[str, tuple]:
    """name -> (p50 us, p90 us, sum us) of each phase's samples.

    q38 L6-INSTR T-1(ii) HP-REPORT: the phases hold different sample counts (a
    phase marked twice per step, or only on some steps), so they are grouped by
    count and each group takes one percentile call and one sum over its rows,
    instead of two percentile calls and a sum per phase. Each value is the float
    the per-phase ``np.percentile(v, 50 / 90)`` and ``v.sum()`` give (the same
    order statistics and interpolation; the same pairwise sum along each
    contiguous row), so the HOSTPHASES line is byte-identical."""
    groups: dict[int, list[str]] = {}
    for name, values in samples.items():
        groups.setdefault(len(values), []).append(name)
    out = {}
    for group in groups.values():
        v = np.asarray([samples[k] for k in group], dtype=np.float64) / 1e3
        q = np.percentile(v, (50, 90), axis=1)
        s = v.sum(axis=1)
        for i, name in enumerate(group):
            out[name] = (q[0, i], q[1, i], s[i])
    return out


def _report() -> None:
    stats = _phase_stats(_samples)
    parts = []
    total = 0.0
    for name in _samples:
        p50, p90, s = stats[name]
        mean = float(s) / _EVERY
        total += mean
        parts.append(f"{name}={p50:.0f}/{p90:.0f}/{mean:.0f}")
    logger.info(
        "HOSTPHASES rank=%d steps=%d per-step host us (p50/p90/mean-per-step) "
        "total_mean=%.0f %s",
        _rank,
        _steps,
        total,
        " ".join(parts),
    )
    _samples.clear()
    if _SUBS:
        _report_subs()
    if _GC:
        _report_gc()


def _report_subs() -> None:
    parts = []
    for name, values in _sub_samples.items():
        v = np.asarray(values, dtype=np.float64) / 1e3
        parts.append(
            f"~{name}={np.percentile(v, 50):.0f}/{np.percentile(v, 90):.0f}/"
            f"{float(v.sum()) / _EVERY:.0f}/{len(values)}"
        )
    logger.info(
        "HOSTSUBS rank=%d steps=%d per-step sub-phase host us "
        "(p50/p90/mean-per-step/n-steps) %s",
        _rank,
        _steps,
        " ".join(parts),
    )
    _sub_samples.clear()


def _gc_cb(phase: str, info: dict) -> None:
    """gc.callbacks timer: one start / stop pair per collection."""
    global _gc_t0
    if phase == "start":
        _gc_t0 = _gc_clock()
    elif _gc_t0 is not None:
        ns = _gc_clock() - _gc_t0
        _gc_t0 = None
        g = min(int(info.get("generation", 0)), 2)
        _gc_win[0] += ns
        _gc_win[1] += 1
        if g == 2:
            _gc_win[2] += 1
        if ns > _gc_win[3]:
            _gc_win[3] = ns
        _gc_gen[g][0] += ns
        _gc_gen[g][1] += 1


def _gc_install() -> None:
    """Append the GC census timer once per process (the worker's first step)."""
    global _gc_hooked
    if _gc_cb not in gc.callbacks:
        gc.callbacks.append(_gc_cb)
    _gc_hooked = True


def _report_gc() -> None:
    w, g = _gc_win, _gc_gen
    logger.info(
        "GCCENSUS rank=%d steps=%d window=%d cyclic GC pauses "
        "(sum_us/collections/gen2_collections) gc_us=%d/%d/%d max_us=%d "
        "g0=%d/%d g1=%d/%d g2=%d/%d",
        _rank,
        _steps,
        _EVERY,
        w[0] // 1000,
        w[1],
        w[2],
        w[3] // 1000,
        g[0][0] // 1000,
        g[0][1],
        g[1][0] // 1000,
        g[1][1],
        g[2][0] // 1000,
        g[2][1],
    )
    w[0] = w[1] = w[2] = w[3] = 0
    for row in g:
        row[0] = row[1] = 0
