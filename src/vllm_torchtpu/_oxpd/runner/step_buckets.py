# SPDX-License-Identifier: Apache-2.0
"""Per-rank counts of model steps by padded token bucket (VLLM_TPU_LOG_STEP_BUCKETS=N).

``count(num_tokens_padded)`` adds one step to that bucket; every N steps the rank logs
the cumulative counts, so a log shows which compiled token buckets a run exercised
(for example, whether short-prompt evals reach the 3,200-token prefill bucket that a
tile change acts on). Host-side only; with N=0 (the default) every call returns at once.

q38 L5 STEPBUCKETS2 (VLLM_TPU_LOG_STEP_BUCKETS_PERIOD=1, needs N > 0): each call also
reads a monotonic host clock, and the period from one call to the next is booked to the
bucket of the earlier call (that step's host cycle). Periods over 50 ms (an idle engine,
a profiler stall) are booked to ``trim`` instead, as R-ROLES trims steps with > 50 ms of
device idle. Every N steps, after the STEPBUCKETS line, the rank logs the cumulative

    STEPBUCKETS2 steps=S periods=P span_us=W trim:n/us <bucket>:n/us ...

where P = S - 1 and every period is booked exactly once, so the bucket sums plus trim
equal span_us (the clock from the first call to the last; each value is floored to
whole microseconds). A period equals the device step only while the host waits for the
device (the draft's dp_entry wait in HOSTPHASES), i.e. while serving is device-bound;
readers print that wait per window. Host-side only: nothing here runs inside a traced
region, and the flag is outside the compile key.
"""

import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_EVERY = envs.VLLM_TPU_LOG_STEP_BUCKETS
_PERIOD = envs.VLLM_TPU_LOG_STEP_BUCKETS_PERIOD
_TRIM_NS = 50_000_000
_counts: dict[int, int] = {}
_steps = 0
_per: dict[int, list[int]] = {}  # bucket -> [periods, sum of periods in ns]
_trim = [0, 0]  # [periods, ns] over _TRIM_NS
_last: list = [None, None, None]  # [clock of the previous call, its bucket, clock of the first call]
_clock = time.perf_counter_ns


def count(num_tokens_padded: int) -> None:
    global _steps
    if not _EVERY:
        return
    n = int(num_tokens_padded)
    if _PERIOD:
        now = _clock()
        prev, bucket = _last[0], _last[1]
        if prev is None:
            _last[2] = now
        else:
            d = now - prev
            if d > _TRIM_NS:
                _trim[0] += 1
                _trim[1] += d
            else:
                acc = _per.setdefault(bucket, [0, 0])
                acc[0] += 1
                acc[1] += d
        _last[0], _last[1] = now, n
    _counts[n] = _counts.get(n, 0) + 1
    _steps += 1
    if _steps % _EVERY == 0:
        _report()


def _report() -> None:
    logger.info(
        "STEPBUCKETS steps=%d %s",
        _steps,
        " ".join(f"{k}:{v}" for k, v in sorted(_counts.items())),
    )
    if _PERIOD:
        span = 0 if _last[2] is None else _last[0] - _last[2]
        logger.info(
            "STEPBUCKETS2 steps=%d periods=%d span_us=%d trim:%d/%d %s",
            _steps,
            sum(v[0] for v in _per.values()) + _trim[0],
            span // 1000,
            _trim[0],
            _trim[1] // 1000,
            " ".join(f"{k}:{v[0]}/{v[1] // 1000}" for k, v in sorted(_per.items())),
        )
