# SPDX-License-Identifier: Apache-2.0
"""Serving-time Python GC stats and an optional delayed gc.disable().

VLLM_TPU_GC_STATS_S=N logs, every N seconds, how many collections ran per
generation and how long they paused the process (total / max ms), plus the
thread they ran on. VLLM_TPU_GC_DISABLE_AFTER_S=T disables the collector T
seconds after warm-up (0 = at warm-up; unset = never), so one run gives the
GC-on and GC-off windows on the same workload.
"""

import gc
import threading
import time

from vllm.logger import init_logger

from vllm_torchtpu import envs

logger = init_logger(__name__)


def install_gc_watch() -> None:
    interval = envs.VLLM_TPU_GC_STATS_S
    disable_after = envs.VLLM_TPU_GC_DISABLE_AFTER_S
    if interval <= 0 and disable_after < 0:
        return
    t_install = time.monotonic()
    state = dict(
        start=0.0,
        window_start=t_install,
        disabled=False,
        count=[0, 0, 0],
        total_ms=[0.0, 0.0, 0.0],
        max_ms=[0.0, 0.0, 0.0],
        over_1ms=0,
        threads={},
    )

    def maybe_disable(now: float) -> None:
        if disable_after >= 0 and not state["disabled"] and now - t_install >= disable_after:
            gc.disable()
            state["disabled"] = True
            logger.info("GC_WATCH gc.disable() at %.0f s after warm-up", now - t_install)

    def flush(now: float) -> None:
        span = now - state["window_start"]
        logger.info(
            "GC_WATCH window=%.0fs gc_enabled=%s count=%s total_ms=%s max_ms=%s "
            "pauses_over_1ms=%d ms_per_s=%.3f threads=%s",
            span,
            gc.isenabled(),
            state["count"],
            [round(x, 1) for x in state["total_ms"]],
            [round(x, 2) for x in state["max_ms"]],
            state["over_1ms"],
            sum(state["total_ms"]) / max(span, 1e-9),
            state["threads"],
        )
        state.update(
            window_start=now,
            count=[0, 0, 0],
            total_ms=[0.0, 0.0, 0.0],
            max_ms=[0.0, 0.0, 0.0],
            over_1ms=0,
            threads={},
        )

    def callback(phase: str, info: dict) -> None:
        if phase == "start":
            state["start"] = time.perf_counter()
            return
        ms = (time.perf_counter() - state["start"]) * 1e3
        g = info.get("generation", 0)
        state["count"][g] += 1
        state["total_ms"][g] += ms
        state["max_ms"][g] = max(state["max_ms"][g], ms)
        state["over_1ms"] += ms > 1.0
        name = threading.current_thread().name
        state["threads"][name] = state["threads"].get(name, 0) + 1
        now = time.monotonic()
        maybe_disable(now)
        if interval > 0 and now - state["window_start"] >= interval:
            flush(now)

    gc.callbacks.append(callback)
    if disable_after == 0:
        maybe_disable(time.monotonic())
    if interval > 0 or disable_after > 0:
        # Collections stop once the collector is disabled (and are rare with a
        # frozen heap), so a timer drives the disable point and the windows.
        def tick() -> None:
            while True:
                time.sleep(max(min(interval if interval > 0 else 5.0, 5.0), 0.5))
                now = time.monotonic()
                maybe_disable(now)
                if interval > 0 and now - state["window_start"] >= interval:
                    flush(now)

        threading.Thread(target=tick, name="gc_watch", daemon=True).start()
