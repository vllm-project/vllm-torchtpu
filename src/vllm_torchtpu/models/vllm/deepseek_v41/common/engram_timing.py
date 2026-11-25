# SPDX-License-Identifier: Apache-2.0
"""Host-side timing of the Engram host-offload path (diagnostic, default off).

TPU_ENGRAM_HOST_TIMING=1 records the wall time of each host stage of a step's
Engram lookup (sync wait for the device hashes, the download, the DP hash
gather, the host row gather plus upload start, the upload activation) and of
the whole model forward on the host, and logs medians/means every
TPU_ENGRAM_HOST_TIMING_EVERY (default 400) forwards. Read straight from the
environment on purpose: the flag must not change the compile-cache hash.
"""

import collections
import contextlib
import os
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("TPU_ENGRAM_HOST_TIMING", "0") == "1"
_EVERY = int(os.environ.get("TPU_ENGRAM_HOST_TIMING_EVERY", "400"))
_samples: dict[str, list[float]] = collections.defaultdict(list)
_forwards = 0


def record(name: str, seconds: float) -> None:
    if ENABLED:
        _samples[name].append(seconds)


@contextlib.contextmanager
def timed(name: str):
    if not ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _samples[name].append(time.perf_counter() - t0)


def forward_done() -> None:
    """Count one forward; log and reset the window every _EVERY forwards."""
    global _forwards
    if not ENABLED:
        return
    _forwards += 1
    if _forwards % _EVERY:
        return
    parts = []
    for name in sorted(_samples):
        v = sorted(_samples[name])
        if not v:
            continue
        parts.append(
            f"{name}: n={len(v)} med={v[len(v) // 2] * 1e3:.2f}ms "
            f"mean={sum(v) / len(v) * 1e3:.2f}ms p90={v[int(0.9 * (len(v) - 1))] * 1e3:.2f}ms"
        )
    logger.info("Engram host timing (%d forwards) | %s", _EVERY, " | ".join(parts))
    _samples.clear()
