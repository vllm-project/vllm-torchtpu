# SPDX-License-Identifier: Apache-2.0
"""Per-step host time of the engine core's phases (VLLM_TPU_ENGINE_PHASES=N).

The runner's HOST_PHASES charges everything between two runner calls to
``idle``. This splits that time on the engine-core side. Every N engine steps
the engine core logs ``ENGINEPHASES``: p50 / p90 / mean microseconds per step
of each phase below, plus the loop period, the running and waiting counts and
the call counts. Any step whose loop period is at least SLOW_MS is also logged
as ``ENGINESLOW`` (with the step before it), with the new requests it
scheduled and their total prompt tokens, so admission steps can be split.

  schedule         scheduler.schedule()
    alloc          kv_cache_manager.allocate_slots() (inside schedule)
    conn_meta      connector.build_connector_meta() (inside schedule)
  exec_dispatch    model_executor.execute_model() (non-blocking: pickling + broadcast)
  sample_dispatch  model_executor.sample_tokens() (non-blocking)
  update           scheduler.update_from_output()
    conn_update    connector.update_connector_output() (inside update)
    free           kv_cache_manager.free()
  wait             the rest of step_fn: mostly waiting for the workers' result
  input            the busy loop's input-queue processing
    add_request    scheduler.add_request() (inside input)
  other            the rest of the loop period (output queue, post_step, ...)
  preproc          preprocess_add_request() on the INPUT THREAD (Request creation,
                   including the prefix-cache block hashing of the whole prompt);
                   not part of the loop period, but it holds the GIL against it

Nested phases are also counted in their parents; ``wait`` and ``other`` are
remainders. With N=0 (the default) nothing is wrapped. Host-side logging only,
so the variable is compile-env-ignored.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

PHASES = (
    "schedule",
    "alloc",
    "conn_meta",
    "exec_dispatch",
    "sample_dispatch",
    "update",
    "conn_update",
    "free",
    "wait",
    "input",
    "add_request",
    "other",
    "preproc",
)
# Phases timed directly inside step_fn; "wait" is step_fn minus their sum.
_IN_STEP = ("schedule", "exec_dispatch", "sample_dispatch", "update")
SLOW_MS = 40.0
MAX_SLOW_PER_WINDOW = 20


class EnginePhases:
    def __init__(self, every: int, core: Any = None, slow_ms: float = SLOW_MS):
        self.every = every
        self.core = core
        self.slow_ns = int(slow_ms * 1e6)
        self.cur = dict.fromkeys(PHASES, 0)
        self.counts = {"add_request": 0, "alloc": 0, "free": 0, "preproc": 0}
        self.samples: dict[str, list[int]] = {p: [] for p in PHASES}
        self.loop: list[int] = []
        self.steps = 0
        self.slow_logged = 0
        self.last_start: int | None = None
        self.prev_len = 0
        self.prev_rec: dict[str, Any] | None = None
        self.step_new = 0
        self.step_new_tokens = 0
        self.step_finished = 0

    # -- wrappers ---------------------------------------------------------
    def timed(self, name: str, fn):
        cur = self.cur
        counts = self.counts

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                cur[name] += time.perf_counter_ns() - t0
                if name in counts:
                    counts[name] += 1

        wrapper.__wrapped__ = fn
        return wrapper

    def timed_schedule(self, fn):
        inner = self.timed("schedule", fn)

        def schedule(*args, **kwargs):
            out = inner(*args, **kwargs)
            try:
                new = getattr(out, "scheduled_new_reqs", None) or ()
                self.step_new += len(new)
                self.step_new_tokens += sum(
                    len(r.prompt_token_ids or ()) for r in new
                )
                self.step_finished += len(getattr(out, "finished_req_ids", None) or ())
            except Exception:  # telemetry must never break scheduling
                pass
            return out

        schedule.__wrapped__ = fn
        return schedule

    def timed_step(self, fn):
        def step(*args, **kwargs):
            t0 = time.perf_counter_ns()
            before = [self.cur[p] for p in _IN_STEP]
            try:
                return fn(*args, **kwargs)
            finally:
                t1 = time.perf_counter_ns()
                inside = sum(self.cur[p] - b for p, b in zip(_IN_STEP, before))
                self.cur["wait"] += max(0, (t1 - t0) - inside)
                self._end_step(t0, t1)

        step.__wrapped__ = fn
        return step

    # -- accounting -------------------------------------------------------
    def _end_step(self, t0: int, t1: int) -> None:
        period = None
        if self.last_start is not None:
            period = t0 - self.last_start
            self.loop.append(period)
            # The previous step's loop period minus its step_fn and the input
            # queue processed since (charged to this step).
            self.cur["other"] += max(0, period - self.prev_len - self.cur["input"])
        rec = {p: self.cur[p] for p in PHASES}
        rec["period"] = period
        rec["new"] = self.step_new
        rec["new_tokens"] = self.step_new_tokens
        rec["finished"] = self.step_finished
        self.prev_len = t1 - t0
        self.last_start = t0
        for p in PHASES:
            self.samples[p].append(self.cur[p])
            self.cur[p] = 0
        self.step_new = self.step_new_tokens = self.step_finished = 0
        self.steps += 1
        if (
            period is not None
            and period >= self.slow_ns
            and self.slow_logged < MAX_SLOW_PER_WINDOW
        ):
            self.slow_logged += 1
            logger.info(
                "ENGINESLOW step=%d period_us=%.0f running=%d %s | prev %s",
                self.steps,
                period / 1e3,
                self._running(),
                self._fmt(rec),
                self._fmt(self.prev_rec) if self.prev_rec else "-",
            )
        self.prev_rec = rec
        if self.steps % self.every == 0:
            self._report()

    @staticmethod
    def _fmt(rec: dict[str, Any]) -> str:
        parts = [f"{p}={rec[p] / 1e3:.0f}" for p in PHASES]
        parts.append(f"new={rec['new']} new_tokens={rec['new_tokens']} finished={rec['finished']}")
        return " ".join(parts)

    def _running(self) -> int:
        sched = getattr(self.core, "scheduler", None)
        try:
            return len(getattr(sched, "running", ()) or ())
        except TypeError:
            return -1

    def _report(self) -> None:
        parts = []
        for p in PHASES:
            v = np.asarray(self.samples[p], dtype=np.float64) / 1e3
            if v.size:
                parts.append(
                    f"{p}={np.percentile(v, 50):.0f}/{np.percentile(v, 90):.0f}/{v.mean():.0f}"
                )
            self.samples[p].clear()
        loop = np.asarray(self.loop, dtype=np.float64) / 1e3
        self.loop.clear()
        sched = getattr(self.core, "scheduler", None)
        try:
            waiting = len(getattr(sched, "waiting", ()) or ())
        except TypeError:
            waiting = -1
        logger.info(
            "ENGINEPHASES steps=%d per-step us (p50/p90/mean-per-step) loop=%s %s "
            "running=%d waiting=%d calls add_request=%d alloc=%d free=%d preproc=%d slow=%d",
            self.steps,
            f"{np.percentile(loop, 50):.0f}/{np.percentile(loop, 90):.0f}/{loop.mean():.0f}"
            if loop.size
            else "-",
            " ".join(parts),
            self._running(),
            waiting,
            self.counts["add_request"],
            self.counts["alloc"],
            self.counts["free"],
            self.counts["preproc"],
            self.slow_logged,
        )
        for k in self.counts:
            self.counts[k] = 0
        self.slow_logged = 0


def install(core: Any, every: int, slow_ms: float = SLOW_MS) -> EnginePhases | None:
    """Wrap one engine core's scheduler, executor and loop methods. Only
    instance attributes change; the classes are untouched."""
    if every <= 0:
        return None
    ph = EnginePhases(every, core, slow_ms)
    sched = core.scheduler
    sched.schedule = ph.timed_schedule(sched.schedule)
    sched.update_from_output = ph.timed("update", sched.update_from_output)
    sched.add_request = ph.timed("add_request", sched.add_request)
    kvm = getattr(sched, "kv_cache_manager", None)
    if kvm is not None:
        kvm.allocate_slots = ph.timed("alloc", kvm.allocate_slots)
        kvm.free = ph.timed("free", kvm.free)
    conn = getattr(sched, "connector", None)
    if conn is not None:
        if hasattr(conn, "build_connector_meta"):
            conn.build_connector_meta = ph.timed("conn_meta", conn.build_connector_meta)
        if hasattr(conn, "update_connector_output"):
            conn.update_connector_output = ph.timed(
                "conn_update", conn.update_connector_output
            )
    ex = core.model_executor
    ex.execute_model = ph.timed("exec_dispatch", ex.execute_model)
    ex.sample_tokens = ph.timed("sample_dispatch", ex.sample_tokens)
    core.step_fn = ph.timed_step(core.step_fn)
    if hasattr(core, "_process_input_queue"):
        core._process_input_queue = ph.timed("input", core._process_input_queue)
    if hasattr(core, "preprocess_add_request"):
        # Called from the input thread; the += on a shared dict can rarely lose
        # a sample against the main thread's reset, which is fine for telemetry.
        core.preprocess_add_request = ph.timed("preproc", core.preprocess_add_request)
    logger.info(
        "Engine phases: on (VLLM_TPU_ENGINE_PHASES=%d, slow >= %.0f ms)", every, slow_ms
    )
    return ph


def patch_engine_core_for_phases(vllm_config: Any = None) -> None:
    """Install the timers when each engine core finishes __init__ (its
    scheduler, executor and step_fn exist then). Called from the engine-core
    process entry point; idempotent; a no-op unless VLLM_TPU_ENGINE_PHASES > 0."""
    every = envs.VLLM_TPU_ENGINE_PHASES
    if every <= 0:
        return
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_engine_phases_patch", False):
        return
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            install(self, every)
        except Exception:  # telemetry must never stop the engine
            logger.exception("Engine phases: install failed; continuing without them")

    EngineCore.__init__ = __init__
    EngineCore._tpu_engine_phases_patch = True
