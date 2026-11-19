# SPDX-License-Identifier: Apache-2.0
"""Sampling profiler for a worker's step thread (VLLM_TPU_PY_SAMPLER_S=N).

At c1 each rank spends ~8 ms of host time per step, of which program
launches are ~1 ms; the rest is Python in the runner. HOSTPHASES says which
phase, not which line. With N > 0, after warmup a daemon thread samples the
step thread's Python stack every VLLM_TPU_PY_SAMPLER_INTERVAL_MS (1 ms) for
N seconds of steps, then logs the top functions by self samples (innermost
frame) and by inclusive samples (anywhere on the stack). Cost: one
``sys._current_frames()`` walk per sample on a different thread.
"""

import collections
import os
import sys
import threading
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _label(frame) -> str:
    code = frame.f_code
    path = code.co_filename
    for marker in ("vllm_torchtpu/", "vllm/", "torch/", "torch_tpu/"):
        i = path.rfind(marker)
        if i >= 0:
            path = path[i:]
            break
    return f"{path}:{code.co_name}"


def _run(target_ident: int, seconds: float, interval: float, rank: int) -> None:
    self_counts: collections.Counter = collections.Counter()
    incl_counts: collections.Counter = collections.Counter()
    samples = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(interval)
        frame = sys._current_frames().get(target_ident)
        if frame is None:
            continue
        samples += 1
        self_counts[_label(frame)] += 1
        seen = set()
        while frame is not None:
            label = _label(frame)
            if label not in seen:
                seen.add(label)
                incl_counts[label] += 1
            frame = frame.f_back
    if not samples:
        return
    def top(counter, n):
        return " | ".join(
            f"{100 * c / samples:.1f}% {name}" for name, c in counter.most_common(n)
        )
    logger.info(
        "PYSAMPLER rank=%d samples=%d interval=%.1fms SELF: %s",
        rank, samples, interval * 1e3, top(self_counts, 30),
    )
    logger.info(
        "PYSAMPLER rank=%d INCLUSIVE: %s", rank, top(incl_counts, 50)
    )


def start(rank: int) -> None:
    """Sample the calling (step) thread; call once after warmup."""
    try:
        seconds = float(os.environ.get("VLLM_TPU_PY_SAMPLER_S", "0") or 0)
        interval = float(os.environ.get("VLLM_TPU_PY_SAMPLER_INTERVAL_MS", "1")) / 1e3
        delay = float(os.environ.get("VLLM_TPU_PY_SAMPLER_DELAY_S", "600"))
    except ValueError:
        return
    if seconds <= 0:
        return
    target = threading.get_ident()

    def delayed():
        # Start once traffic is steady (the benchmark warms up after startup).
        time.sleep(delay)
        _run(target, seconds, interval, rank)

    threading.Thread(target=delayed, name="py-sampler", daemon=True).start()
    logger.info(
        "Python sampler armed: rank %d, %.0f s after a %.0f s delay, every %.1f ms",
        rank, seconds, delay, interval * 1e3,
    )
