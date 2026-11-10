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
"""Small-batch decode host-overhead trims (IX track 3). Every switch is
default-off and read from ``os.environ`` (not ``envs.py``, so the
compile-cache key is unchanged). None changes what the device computes.
``VLLM_TPU_T3_HOST=1`` turns every trim on; an explicitly set per-trim
variable (``=0`` or ``=1``) wins over it.

* ``VLLM_TPU_T3_AOT_SKIP_GUARDS=1``: once a vLLM ``@support_torch_compile``
  module (here the MTP draft step-0 and fused-loop programs) has run its
  freshly AOT-compiled function, skip that function's per-call Dynamo guard
  evaluation, exactly as vLLM already does for an AOT function it *loads*
  from cache when ``dynamic_shapes_config.evaluate_guards`` is False
  (``vllm/compilation/decorators.py:_try_load_aot_compiled_fn``). The
  compiled callable that runs is the same object either way.
* ``VLLM_TPU_T3_GUARD_MEMO``: see ``vllm_torchtpu._qx_fastcall``.
"""

from __future__ import annotations

import os
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _flag(name: str) -> bool:
    """``name`` if set, else the master switch ``VLLM_TPU_T3_HOST``."""
    raw = os.environ.get(name, os.environ.get("VLLM_TPU_T3_HOST", "0"))
    return raw.strip().lower() in ("1", "true", "on", "yes")


AOT_SKIP_GUARDS = _flag("VLLM_TPU_T3_AOT_SKIP_GUARDS")

# VLLM_TPU_T3_CONN_STATS_EVERY=N: poll the KV connector's stats every N decode
# steps instead of every step (default 1 = every step; 16 under
# VLLM_TPU_T3_HOST=1 unless set explicitly).
CONN_STATS_EVERY = int(
    os.environ.get(
        "VLLM_TPU_T3_CONN_STATS_EVERY",
        "16" if _flag("VLLM_TPU_T3_CONN_STATS") else "1",
    )
    or 1
)

_done: set[int] = set()


def _evaluate_guards(module: Any) -> bool:
    """vLLM's evaluate_guards for this module (True when unknown: keep guards)."""
    cfg = getattr(module, "compilation_config", None)
    if cfg is None:
        vcfg = getattr(module, "vllm_config", None)
        cfg = getattr(vcfg, "compilation_config", None)
    ds = getattr(cfg, "dynamic_shapes_config", None)
    if ds is None:
        return True
    return bool(getattr(ds, "evaluate_guards", True))


def maybe_skip_aot_guards(module: Any) -> bool:
    """After a successful call of ``module``: disable its AOT function's guard
    check when vLLM would have done so on a cache load. Returns True when it
    was disabled by this call. Cheap after the first call per module."""
    if not AOT_SKIP_GUARDS:
        return False
    key = id(module)
    if key in _done:
        return False
    fn = getattr(module, "aot_compiled_fn", None)
    if fn is None:
        return False
    _done.add(key)
    if not getattr(fn, "_guard_check_enabled", False):
        return False
    if _evaluate_guards(module):
        logger.info(
            "VLLM_TPU_T3_AOT_SKIP_GUARDS: %s keeps guards (evaluate_guards=True)",
            type(module).__name__,
        )
        return False
    disable = getattr(fn, "disable_guard_check", None)
    if disable is None:
        return False
    disable()
    logger.info(
        "VLLM_TPU_T3_AOT_SKIP_GUARDS: per-call guard check off for %s "
        "(as vLLM's AOT cache-load path does with evaluate_guards=False)",
        type(module).__name__,
    )
    return True


def _reset_for_tests() -> None:
    _done.clear()


# ---------------------------------------------------------------------------
# Per-step process knobs (default off), applied from the worker's
# execute_model and the engine core's schedule:
#   VLLM_TPU_T3_SWITCH_US=N : sys.setswitchinterval(N us) instead of Python's
#                             5 ms default, so a background Python thread
#                             (output / connector / submit threads) cannot hold
#                             the GIL for long while the step loop waits on it.
#   VLLM_TPU_T3_GC=freeze   : after VLLM_TPU_T3_GC_AFTER steps (default 500):
#                             gc.collect(); gc.freeze(); and raise the gen-0
#                             threshold to VLLM_TPU_T3_GC_GEN0 (default 50000),
#                             so cyclic GC runs rarely in the decode loop.
# ---------------------------------------------------------------------------

import gc as _gc
import sys as _sys

SWITCH_US = float(os.environ.get("VLLM_TPU_T3_SWITCH_US", "0") or 0)
GC_MODE = os.environ.get("VLLM_TPU_T3_GC", "").strip().lower()
_GC_AFTER = int(os.environ.get("VLLM_TPU_T3_GC_AFTER", "500") or 500)
_GC_GEN0 = int(os.environ.get("VLLM_TPU_T3_GC_GEN0", "50000") or 50000)
STEP_HOOK = bool(SWITCH_US > 0 or GC_MODE == "freeze")
_DEFAULT_SWITCH = _sys.getswitchinterval()
_DEFAULT_GC_THRESH = _gc.get_threshold()
_hook = {"n": 0, "applied": False, "frozen": False}


def _apply() -> None:
    if SWITCH_US > 0:
        _sys.setswitchinterval(SWITCH_US / 1e6)
    if GC_MODE == "freeze" and _hook["frozen"]:
        _gc.set_threshold(_GC_GEN0, _DEFAULT_GC_THRESH[1], _DEFAULT_GC_THRESH[2])


def on_step() -> None:
    """Cheap per-step hook; only acts when a knob is set."""
    _hook["n"] += 1
    if GC_MODE == "freeze" and not _hook["frozen"] and _hook["n"] >= _GC_AFTER:
        _gc.collect()
        _gc.freeze()
        _hook["frozen"] = True
        _hook["applied"] = False  # re-apply thresholds below
        logger.info(
            "VLLM_TPU_T3_GC=freeze: froze %d objects after %d steps; gen-0 threshold %d",
            _gc.get_freeze_count(), _hook["n"], _GC_GEN0,
        )
    if not _hook["applied"]:
        _hook["applied"] = True
        _apply()
        if _hook["n"] < 3:
            logger.info(
                "VLLM_TPU_T3 step knobs: switch_us=%s gc=%s (pid %d)",
                SWITCH_US, GC_MODE or "default", os.getpid(),
            )
