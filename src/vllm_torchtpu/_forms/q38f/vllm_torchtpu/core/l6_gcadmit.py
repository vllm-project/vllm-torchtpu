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
"""q38 L6 GC-ADMIT-FREEZE: lever ``gcadmit`` (``VLLM_TPU_L6_GC_ADMIT_FREEZE``; worker process; default off).

Why. On T98o (j-ff638d81, M) every decode rank ran a gen-2 cyclic collection of 13.1-19.4 s in the
long-context burn-in. vLLM's ``freeze_gc_heap()`` (worker/tpu_worker.py, end of warm-up) runs once, and
D1e's ``VLLM_TPU_HOST_GC=freeze`` is one-shot too, so request state created later stays in the collected
heap and every gen-2 collection walks it again (a request's prompt token-id list holds one reference per
token when the worker has the whole prompt).

What. After a worker step that admitted requests (``scheduler_output.scheduled_new_reqs`` is not empty),
right after ``_update_states`` has built their state, the lever runs, once per such step:

1. ``gc.collect()``. Cyclic garbage that is already unreachable is reclaimed, so the freeze cannot pin it.
   Every earlier admission froze whatever was live then, so this walks only objects created since the
   previous freeze (plus the new requests' state, once).
2. ``gc.freeze()``. Every live tracked object, the new requests' state included, moves to the permanent
   generation, which later collections skip.

It never calls ``gc.unfreeze()``: that would also thaw the warm-up ``freeze_gc_heap()`` set. Frozen objects
are still freed by reference counting when they die (a finished request's state goes away as before); only
an object that is part of a reference cycle when it dies stays allocated (frozen cyclic garbage).

Cap. Frozen cyclic garbage cannot be reclaimed, so its growth is bounded. Each freeze measures what it moves:
after the full collection every surviving non-frozen tracked object is in generation 2, so
``m = len(gc.get_objects(generation=2))``, read between the collection and the freeze, is the number of objects
this freeze moves (O(objects created since the previous freeze)). The lever keeps ``cum``, the sum of m over its
freezes, and stops freezing for the rest of the run (one WARNING; nothing is unfrozen) once
``VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP`` (default 1000000; 0 = no cap) is set and ``cum`` exceeds it. Frozen objects
leave the permanent generation only by dying, so ``cum`` is an upper bound on its growth since the first freeze.
The cap is conservative: legitimate churn (a request's state frozen at admission and freed when it finishes)
counts toward ``cum`` too, so a long run with many admissions reaches the cap after about CAP / (mean m)
freezes even when nothing leaks, and from then on behaves like the lever being off. ``gc.get_freeze_count()``
is never called: in CPython 3.12 it walks the whole permanent generation (117-132 ms at 1.64 M frozen objects
on the CDK CPU pod, M: j-698978c0).

Gates. Engages only after warm-up (``runner.warmup_done``, which the worker sets after ``freeze_gc_heap()``).
Refused (one ERROR, lever off for the run) next to D1e's ``VLLM_TPU_HOST_GC=freeze`` (it owns the freeze) and
next to the L6 ``gcfreeze`` lever (``VLLM_TPU_L6_GC_FREEZE``, which calls ``gc.unfreeze()`` at every off
window). Both are read from the environment directly. Any exception turns the lever off with one ERROR.

Values: ``0`` / ``1`` / ``ab:<classes>`` (core/l6_ab.py). ``ab:`` reads ``l6_ab.on("gcadmit")``, so it needs the
worker's ``l6_ab.begin_step``. A tree whose worker never calls it is detected at the first admission
(``l6_ab.step_seq()`` is 0) and ``ab:`` is refused with one ERROR. An ``ab:`` contrast is not clean: what an on
window froze stays frozen in the off windows after it.

Counts (``l6_ab.note``, in the L6AB window line where the worker logs one): ``gcadmit.freeze``,
``gcadmit.us`` (µs in collect + freeze), ``gcadmit.reclaimed`` (objects the pre-freeze collection found
unreachable), ``gcadmit.n_frozen`` (objects moved by its freezes, the sum of m), ``gcadmit.capped``.
Its own line, at the first three freezes, at every ``VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK``-th freeze (default 64)
and at the freeze that reaches the cap:
``q38 L6 GCADMIT freeze=<n> admitted=<k> us=<t> reclaimed=<r> frozen_this=<m> frozen_cum=<cum>``.
A freeze's collection is a full (generation 2) collection, so GC timers that count collections per
generation (GCCENSUS, D1e, SYNC-GC, GCSLOW) count one more gen-2 collection per admission step.
"""

from __future__ import annotations

import gc
import os
import time

from vllm_torchtpu.logger import init_logger

from . import l6_ab

logger = init_logger(__name__)

ENV = "VLLM_TPU_L6_GC_ADMIT_FREEZE"
_OFF = ("", "0", "false", "off")
_UNIFORM = ("1", "true", "on")

# (Re)built by _resolve(); tests set os.environ and call it.
ON = False
UNIFORM = False
CAP = 1000000
CHECK = 64
_st: dict = {}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    v = int(raw) if raw else default
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def _resolve() -> None:
    global ON, UNIFORM, CAP, CHECK
    raw = os.environ.get(ENV, "").strip().lower()
    if raw not in _OFF and raw not in _UNIFORM and not raw.startswith("ab:"):
        raise ValueError(f"{ENV}={raw!r}: expected 0, 1 or ab:<c>[,<c>...]")
    UNIFORM = raw in _UNIFORM
    ON = UNIFORM or raw.startswith("ab:")
    CAP = _env_int("VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP", 1000000)
    CHECK = _env_int("VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK", 64)
    if CHECK < 1:
        raise ValueError("VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK must be >= 1")
    _st.clear()
    _st.update(checked=False, off=False, n=0, cum=0)


_resolve()


def _refusal() -> str | None:
    if os.environ.get("VLLM_TPU_HOST_GC", "").strip().lower() == "freeze":
        return "D1e's VLLM_TPU_HOST_GC=freeze owns the freeze"
    if os.environ.get("VLLM_TPU_L6_GC_FREEZE", "").strip().lower() not in _OFF:
        return "the gcfreeze lever (VLLM_TPU_L6_GC_FREEZE) calls gc.unfreeze() at every off window"
    return None


def _off(why: str) -> None:
    _st["off"] = True
    logger.error("q38 L6 gcadmit off for the rest of the run: %s", why)


def _on() -> bool:
    if UNIFORM:
        return True
    if l6_ab.step_seq() == 0:
        _off("ab: windows need the worker's l6_ab.begin_step, which this worker never called")
        return False
    return l6_ab.on("gcadmit")


def _freeze(admitted: int) -> None:
    t0 = time.perf_counter_ns()
    reclaimed = gc.collect()
    m = len(gc.get_objects(generation=2))  # what the freeze moves: every survivor is in generation 2 now
    gc.freeze()
    us = (time.perf_counter_ns() - t0) // 1000
    _st["n"] += 1
    _st["cum"] += m
    n, cum = _st["n"], _st["cum"]
    l6_ab.note("gcadmit", "freeze")
    l6_ab.note("gcadmit", "us", us)
    l6_ab.note("gcadmit", "reclaimed", reclaimed)
    l6_ab.note("gcadmit", "n_frozen", m)
    capped = bool(CAP) and cum > CAP
    if n <= 3 or n % CHECK == 0 or capped:
        logger.info("q38 L6 GCADMIT freeze=%d admitted=%d us=%d reclaimed=%d frozen_this=%d frozen_cum=%d",
                    n, admitted, us, reclaimed, m, cum)
    if capped:
        _st["off"] = True
        l6_ab.note("gcadmit", "capped")
        logger.warning(
            "q38 L6 gcadmit capped after %d freezes: %d objects frozen in total (> VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP"
            "=%d); no further freezes, nothing unfrozen", n, cum, CAP)


def after_update_states(runner, scheduler_output) -> None:
    """Called by the worker's execute_model right after ``_update_states`` (only when ``ON``)."""
    if not ON or _st["off"]:
        return
    try:
        if not _st["checked"]:
            _st["checked"] = True
            why = _refusal()
            if why:
                _st["off"] = True
                logger.error("q38 L6 gcadmit refused: %s; lever off.", why)
                return
            logger.info("q38 L6 gcadmit armed: %s=%s cap=%d check=%d (pid %d)", ENV,
                        os.environ.get(ENV, "").strip(), CAP, CHECK, os.getpid())
        if not getattr(runner, "warmup_done", False):
            return
        new = scheduler_output.scheduled_new_reqs
        if not new:
            return
        if not _on():
            return
        _freeze(len(new))
    except Exception as exc:  # noqa: BLE001 - an optimisation must never stop the step
        _off(f"{type(exc).__name__}: {exc}")
