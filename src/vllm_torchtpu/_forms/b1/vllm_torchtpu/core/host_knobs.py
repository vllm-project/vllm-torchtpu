# SPDX-License-Identifier: Apache-2.0
"""q38 SV-KNOB GIL-SW: the interpreter's thread switch interval for the decode loop, with in-process A/B windows.

Default off. A port onto K3 of the oxpd form's ``core/host_knobs.py`` (q38 D1e, 74c03159ab), itself a port of the
per-step knobs and in-process A/B windows of IX's track-3 host trims (``runner/t3_host.py``, Qwen3.5 lane; both
credited). Only the switch interval and the A/B windows are ported. Nothing here changes what the device computes:
the module calls ``sys.setswitchinterval`` and logs, nothing else.

* ``VLLM_TPU_HOST_SWITCH_US=N`` (integer microseconds, 1..100000; unset or 0 = off): ``sys.setswitchinterval(N us)``
  instead of the interpreter's default (5 ms), so a thread holding the GIL (the engine core's input thread hashing a
  new prompt, a worker's async-output thread) is asked to hand it to the main loop sooner.
* ``VLLM_TPU_HOST_AB_PERIOD_MS=P`` (unset or 0 = off): in-process A/B. The interval is N in even wall-clock windows
  (``time.time_ns() // (P ms)`` even) and the interpreter's default in odd ones. P must divide 86,400,000 with an
  even quotient (20000 does: 4320), so a window's parity is fixed by the UTC time of day and is the same in every
  process on every host. With N unset the windows are still logged and nothing is toggled (an A/A control).
* Garbage collection is not ported. K3 sets its GC policy with ``VLLM_TPU_GC_*`` (``runner/gc_at_waits.py``); a
  second mechanism could fight it, so the oxpd keys ``VLLM_TPU_HOST_GC``, ``VLLM_TPU_HOST_GC_AFTER`` and
  ``VLLM_TPU_HOST_GC_GEN0`` are refused when set. This module never imports ``gc``.

Refusals raise ``ValueError`` when the module is imported, as K3's env parsers do: a value that is not a
non-negative integer, a switch interval above 100000 us, an A/B period that does not divide 86,400,000 with an
even quotient, and any of the oxpd GC keys set.

Each process (the engine core through ``scheduler.schedule``, every worker through ``execute_model``) applies the
knob at its own first step and logs, with its own wall clock:
  ``Host knobs (q38 SV-KNOB): role=.. pid=.. switch_us=.. ab_period_ms=.. default_switch_s=..`` once, and
  ``Host knobs AB (q38 SV-KNOB): role=.. pid=.. window=k on=0|1 t_ns=.. prev_window=.. steps=.. switch_s=..``
at every window change (``steps`` = this process's steps in the previous window; ``switch_s`` = the interval now).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DAY_MS = 86_400_000
SWITCH_US_MAX = 100_000
REFUSED_GC_KEYS = ("VLLM_TPU_HOST_GC", "VLLM_TPU_HOST_GC_AFTER", "VLLM_TPU_HOST_GC_GEN0")

SWITCH_US = 0
AB_PERIOD_MS = 0
ENABLED = False
_PERIOD_NS = 0
_DEFAULT_SWITCH = sys.getswitchinterval()


def _resolve() -> None:
    """Validates the keys (raises ValueError) and sets ENABLED and the window length."""
    global ENABLED, _PERIOD_NS
    for key in REFUSED_GC_KEYS:
        if os.environ.get(key, "").strip():
            raise ValueError(
                f"{key} is refused on this tree: host_knobs ports only the switch interval and the A/B windows; "
                "the GC policy is VLLM_TPU_GC_* (runner/gc_at_waits.py)."
            )
    if not isinstance(SWITCH_US, int) or not 0 <= SWITCH_US <= SWITCH_US_MAX:
        raise ValueError(
            f"Invalid value '{SWITCH_US}' for VLLM_TPU_HOST_SWITCH_US: must be an integer in 0..{SWITCH_US_MAX}."
        )
    if not isinstance(AB_PERIOD_MS, int) or AB_PERIOD_MS < 0 or (
        AB_PERIOD_MS > 0 and (DAY_MS % AB_PERIOD_MS != 0 or (DAY_MS // AB_PERIOD_MS) % 2 != 0)
    ):
        raise ValueError(
            f"Invalid value '{AB_PERIOD_MS}' for VLLM_TPU_HOST_AB_PERIOD_MS: must be 0 or divide {DAY_MS} with an "
            "even quotient (e.g. 20000)."
        )
    ENABLED = SWITCH_US > 0 or AB_PERIOD_MS > 0
    _PERIOD_NS = AB_PERIOD_MS * 1_000_000


def _load() -> None:
    """Reads the keys (at import; tests call it again after changing the environment)."""
    global SWITCH_US, AB_PERIOD_MS, ENABLED
    ENABLED = False
    SWITCH_US = envs.VLLM_TPU_HOST_SWITCH_US
    AB_PERIOD_MS = envs.VLLM_TPU_HOST_AB_PERIOD_MS
    _resolve()


_load()


def _now_ns() -> int:
    return time.time_ns()


_state: dict[str, Any] = {}


def _reset_for_tests() -> None:
    _state.clear()
    _state.update(started=False, win=None, phase=None, steps=0, n=0, windows=[])


_reset_for_tests()


def _apply(on: bool) -> None:
    if SWITCH_US > 0:
        sys.setswitchinterval(SWITCH_US / 1e6 if on else _DEFAULT_SWITCH)
    _state["phase"] = on


def _window_change(role: str, k: int, now: int) -> None:
    prev, steps = _state["win"], _state["steps"]
    on = k % 2 == 0
    _state["win"], _state["steps"] = k, 0
    _state["windows"].append((k, on))
    _apply(on)
    logger.info(
        "Host knobs AB (q38 SV-KNOB): role=%s pid=%d window=%d on=%d t_ns=%d prev_window=%s steps=%d switch_s=%.6f",
        role, os.getpid(), k, int(on), now, prev, steps, sys.getswitchinterval(),
    )


def on_step(role: str) -> None:
    """Per-step hook; callers guard it with ``ENABLED``."""
    if not _state["started"]:
        _state["started"] = True
        logger.info(
            "Host knobs (q38 SV-KNOB): role=%s pid=%d switch_us=%d ab_period_ms=%d default_switch_s=%.6f",
            role, os.getpid(), SWITCH_US, AB_PERIOD_MS, _DEFAULT_SWITCH,
        )
    if _PERIOD_NS:
        now = _now_ns()
        k = now // _PERIOD_NS
        if k != _state["win"]:
            _window_change(role, k, now)
    elif _state["phase"] is not True:
        _apply(True)
    _state["steps"] += 1
    _state["n"] += 1


def install_engine_core(core: Any) -> bool:
    """Wraps one engine core's ``scheduler.schedule`` (instance attribute only)."""
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
    """Installs the engine-core hook when each engine core finishes ``__init__``.

    Called from the engine-core process entry point; idempotent; a no-op (nothing imported, nothing patched)
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
            logger.exception("Host knobs (q38 SV-KNOB): engine-core install failed; continuing without them")

    EngineCore.__init__ = __init__
    EngineCore._q38_host_knobs_patch = True
    return True
