# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""q38 D1e: per-process interpreter knobs for the decode loop (default off).

A port of the per-step knobs and the in-process A/B windows of IX's track-3
host trims (``runner/t3_host.py``, Qwen3.5 lane, credited). None changes what
the device computes; they only change when the Python interpreter switches
threads and runs its cyclic garbage collector, which shows up as random
1-2.5 ms host stalls on one rank (levers/decode.md D1e).

* ``VLLM_TPU_HOST_SWITCH_US=N``: ``sys.setswitchinterval(N µs)`` instead of
  Python's 5 ms default.
* ``VLLM_TPU_HOST_GC=freeze``: after ``VLLM_TPU_HOST_GC_AFTER`` steps (500):
  ``gc.collect(); gc.freeze()`` and the gen-0 threshold raised to
  ``VLLM_TPU_HOST_GC_GEN0`` (50000). One-shot.
* ``VLLM_TPU_HOST_GC=threshold``: only the gen-0 threshold raise, after the
  same step count (stateless, so it can be windowed).
* ``VLLM_TPU_HOST_AB_PERIOD_MS=P`` (0 = off): in-process A/B. The knobs are on
  in even wall-clock windows (``time_ns // P ms`` even) and back at the
  interpreter defaults in odd ones; P should divide 86,400,000 with an even
  quotient (e.g. 20000) so window parity is fixed by the time of day. Each
  process logs every window change with that window's GC pause time per
  generation (``gc.callbacks``). ``gc.freeze`` is one-shot and cannot be
  windowed, so AB mode refuses ``VLLM_TPU_HOST_GC=freeze`` (all knobs off).

The GC timer is installed at each process's first step whatever the AB period
(q38 L6 SYNC-GC, reports/host_levers_h1.md T-5(b): served runs use threshold mode
without AB and so never measured GC). Without AB, each process logs its GC pauses
per generation every 20 s on a ``Host knobs GC (q38 D1e)`` line.

Applied from the worker's ``execute_model`` and the engine core's
``schedule`` (each process counts its own steps).
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Read at import; tests set the module globals and call _resolve().
SWITCH_US = float(envs.VLLM_TPU_HOST_SWITCH_US or 0)
GC_MODE = (envs.VLLM_TPU_HOST_GC or "").strip().lower()
GC_AFTER = int(envs.VLLM_TPU_HOST_GC_AFTER)
GC_GEN0 = int(envs.VLLM_TPU_HOST_GC_GEN0)
AB_PERIOD_MS = int(envs.VLLM_TPU_HOST_AB_PERIOD_MS)
ENABLED = False
AB_REFUSED = False
_DEFAULT_SWITCH = sys.getswitchinterval()
_DEFAULT_GEN0 = gc.get_threshold()[0]


def _resolve() -> None:
    global ENABLED, AB_REFUSED
    AB_REFUSED = AB_PERIOD_MS > 0 and GC_MODE == "freeze"
    ENABLED = bool(SWITCH_US > 0 or GC_MODE in ("freeze", "threshold")) and not AB_REFUSED
    if AB_REFUSED:
        logger.error(
            "Host knobs (q38 D1e): VLLM_TPU_HOST_AB_PERIOD_MS=%d cannot window gc.freeze (one-shot); "
            "use VLLM_TPU_HOST_GC=threshold. All host knobs stay off.", AB_PERIOD_MS)


_resolve()


def _now_ns() -> int:
    return time.time_ns()


_state: dict[str, Any] = {}
_gc = {"t0": 0, "ms": [0.0, 0.0, 0.0], "n": [0, 0, 0], "hooked": False}
_GC_LOG_NS = 20_000_000_000  # the GC line period without AB windows


def _reset_for_tests() -> None:
    _state.clear()
    _state.update(n=0, applied=False, gc_ready=False, frozen=False, phase=None, win=None, steps=0,
                  windows=[], last_stats=None, gc_log_t=None)
    _gc.update(ms=[0.0, 0.0, 0.0], n=[0, 0, 0])


_reset_for_tests()


def _gc_cb(phase: str, info: dict) -> None:
    if phase == "start":
        _gc["t0"] = time.perf_counter_ns()
    elif _gc["t0"]:
        g = min(int(info.get("generation", 0)), 2)
        _gc["ms"][g] += (time.perf_counter_ns() - _gc["t0"]) / 1e6
        _gc["n"][g] += 1
        _gc["t0"] = 0


def _apply(on: bool) -> None:
    if SWITCH_US > 0:
        sys.setswitchinterval(SWITCH_US / 1e6 if on else _DEFAULT_SWITCH)
    if _state["gc_ready"] and GC_MODE in ("freeze", "threshold"):
        _, g1, g2 = gc.get_threshold()
        gc.set_threshold(GC_GEN0 if on else _DEFAULT_GEN0, g1, g2)


def _window_change(role: str, k: int) -> None:
    prev, on = _state["win"], k % 2 == 0
    stats = {"window": prev, "on": None if prev is None else prev % 2 == 0, "steps": _state["steps"],
             "gc_ms": [round(x, 3) for x in _gc["ms"]], "gc_n": list(_gc["n"])}
    _state["last_stats"] = stats
    _state["win"], _state["steps"] = k, 0
    _gc.update(ms=[0.0, 0.0, 0.0], n=[0, 0, 0])
    _state["windows"].append((k, on))
    _state["phase"] = on
    _apply(on)
    logger.info(
        "Host knobs AB (q38 D1e): role=%s window=%d on=%d prev_window=%s steps=%d gc_ms=%s gc_n=%s",
        role, k, int(on), prev, stats["steps"], stats["gc_ms"], stats["gc_n"])


def on_step(role: str) -> None:
    """Cheap per-step hook; callers guard it with ``ENABLED``."""
    _state["n"] += 1
    _state["steps"] += 1
    if not _state["applied"]:
        _state["applied"] = True
        if not _gc["hooked"]:  # whatever AB_PERIOD_MS (q38 L6 SYNC-GC)
            gc.callbacks.append(_gc_cb)
            _gc["hooked"] = True
        logger.info(
            "Host knobs (q38 D1e): role=%s switch_us=%s gc=%s ab_period_ms=%d (pid %d)",
            role, SWITCH_US, GC_MODE or "default", AB_PERIOD_MS, os.getpid(),
        )
    if GC_MODE in ("freeze", "threshold") and not _state["gc_ready"] and _state["n"] >= GC_AFTER:
        if GC_MODE == "freeze":
            gc.collect()
            gc.freeze()
            _state["frozen"] = True
            logger.info(
                "Host knobs (q38 D1e): role=%s gc.freeze froze %d objects after %d steps; gen-0 threshold %d",
                role, gc.get_freeze_count(), _state["n"], GC_GEN0,
            )
        _state["gc_ready"] = True
        _state["phase"] = None  # re-apply below with the threshold
    if AB_PERIOD_MS > 0:
        k = _now_ns() // (AB_PERIOD_MS * 1_000_000)
        if k != _state["win"]:
            _window_change(role, k)
        elif _state["phase"] is None:
            _state["phase"] = k % 2 == 0
            _apply(_state["phase"])
    else:
        if _state["phase"] is not True:
            _state["phase"] = True
            _apply(True)
        _gc_line(role)


def _gc_line(role: str) -> None:
    """Without AB windows: this process's GC pauses per generation every 20 s."""
    now = _now_ns()
    if _state["gc_log_t"] is None:
        _state["gc_log_t"] = now
        return
    if now - _state["gc_log_t"] < _GC_LOG_NS:
        return
    logger.info(
        "Host knobs GC (q38 D1e): role=%s steps=%d secs=%.1f gc_ms=%s gc_n=%s",
        role, _state["steps"], (now - _state["gc_log_t"]) / 1e9, [round(x, 3) for x in _gc["ms"]], list(_gc["n"]))
    _state["gc_log_t"], _state["steps"] = now, 0
    _gc.update(ms=[0.0, 0.0, 0.0], n=[0, 0, 0])


def install_engine_core(core: Any) -> bool:
    """Wrap one engine core's ``scheduler.schedule`` (instance attribute only)."""
    sched = getattr(core, "scheduler", None)
    if sched is None or getattr(sched.schedule, "_q38_host_knobs", False):
        return False
    orig = sched.schedule

    def schedule(*args, **kwargs):
        on_step("engine_core")
        return orig(*args, **kwargs)

    schedule._q38_host_knobs = True  # type: ignore[attr-defined]
    schedule.__wrapped__ = orig  # type: ignore[attr-defined]
    sched.schedule = schedule
    return True


def patch_engine_core_for_host_knobs() -> bool:
    """Install the engine-core hook when each engine core finishes __init__.
    Called from the engine-core process entry point; idempotent; a no-op
    unless a knob is set."""
    if not ENABLED:
        return False
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_q38_host_knobs_patch", False):
        return True
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            install_engine_core(self)
        except Exception:  # an optimisation must never stop the engine
            logger.exception("Host knobs: engine-core install failed; continuing without them")

    EngineCore.__init__ = __init__
    EngineCore._q38_host_knobs_patch = True
    return True
