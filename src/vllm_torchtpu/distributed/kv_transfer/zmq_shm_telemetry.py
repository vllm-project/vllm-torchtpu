# SPDX-License-Identifier: Apache-2.0
"""Records latency by KV transfer phase."""

import threading
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger("vllm_torchtpu.distributed.kv_transfer.zmq_shm_base")

# Phases tracked by _LatencyTracker. Kept as module-level strings so call
# sites don't typo-drift.
_LAT_D2H = "d2h"  # producer: device -> host (index_select + copy_)
_LAT_WIRE = "wire"  # consumer: network PULL round-trip
_LAT_UNPACK = "unpack"  # consumer: rank0 scatter payload -> shm
_LAT_H2D = "h2d"  # consumer: host -> device (to(device))
_LAT_INSERT = "insert"  # consumer: index_put_ + synchronize on device
_LAT_STAGE = "stage"  # producer: total staging (D2H across all layers+sync)
_LAT_SCATTER = "scatter"  # consumer: total scatter (H2D+insert across layers)


class _LatencyTracker:
    """Running per-phase latency stats + periodic summary log.

    Phases are arbitrary string keys (see _LAT_* module constants). Every
    record() updates n/total/min/max under a lock; the first record() after
    log_interval seconds triggers a summary line. Per-sample lines are up
    to the caller (they often carry req-specific context we'd lose here)."""

    def __init__(self, who: str, log_interval_s: float = 30.0):
        self._who = who
        self._log_interval_s = max(0.0, log_interval_s)
        self._lock = threading.Lock()
        # phase -> [n, total_ms, min_ms, max_ms]
        self._stats: dict[str, list[float]] = {}
        self._last_log = time.perf_counter()

    def record(self, phase: str, duration_ms: float) -> None:
        with self._lock:
            row = self._stats.get(phase)
            if row is None:
                self._stats[phase] = [
                    1.0, duration_ms, duration_ms, duration_ms
                ]
            else:
                row[0] += 1
                row[1] += duration_ms
                if duration_ms < row[2]:
                    row[2] = duration_ms
                if duration_ms > row[3]:
                    row[3] = duration_ms
            if self._log_interval_s <= 0:
                return
            now = time.perf_counter()
            if now - self._last_log < self._log_interval_s:
                return
            parts = []
            for name, (n, total, mn, mx) in self._stats.items():
                avg = total / n if n else 0.0
                parts.append(f"{name}: n={int(n)} avg={avg:.1f}ms "
                             f"min={mn:.1f}ms max={mx:.1f}ms")
            self._last_log = now
        logger.info("%s latency summary | %s", self._who, " | ".join(parts))
