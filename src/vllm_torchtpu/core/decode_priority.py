# SPDX-License-Identifier: Apache-2.0
"""Decode priority after a large prefill step (VLLM_TPU_DECODE_PRIORITY=N:K).

At c32 the p90-interactivity requests spend ~57% of their decode window inside
mixed steps carrying an 8K prefill chunk (j-aff13101 / j-63f072c8), and the
scheduler follows such a step with about one decode step before the next chunk.
With this flag, after any step that schedules >= N prefill tokens the next K
steps defer prefill work (in-progress chunks and new admissions) and run the
running decodes only, using vLLM's own DP prefill-balancing path
(`schedule(throttle_prefills=True)`). Prefill chunk sizes are unchanged; only
their spacing is.

A step's prefill tokens are its requests' scheduled tokens above the decode
size (1 + speculative tokens). The window is skipped when nothing is decoding
(vLLM's `defer_prefills` then admits prefills anyway).
"""

import functools
import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _parse() -> tuple[int, int] | None:
    raw = os.environ.get("VLLM_TPU_DECODE_PRIORITY", "").strip()
    if not raw:
        return None
    n, _, k = raw.partition(":")
    n, k = int(n), int(k)
    if n <= 0 or k <= 0:
        raise ValueError(f"VLLM_TPU_DECODE_PRIORITY={raw!r}: want N:K with N, K > 0")
    return n, k


def patch_decode_priority() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    cfg = _parse()
    if cfg is None or getattr(Scheduler.schedule, "_tpu_decode_priority", False):
        return
    threshold, window = cfg
    original = Scheduler.schedule

    @functools.wraps(original)
    def schedule(self, throttle_prefills: bool = False):
        left = getattr(self, "_tpu_decode_only_left", 0)
        if left > 0:
            self._tpu_decode_only_left = left - 1
            throttle_prefills = True
            # vLLM releases deferral when the waiting queue outgrows the deferred
            # set (DP capacity rule); here the window is the whole point.
            self.prefill_capacity_bound = False
        out = original(self, throttle_prefills)
        if left <= 0:
            spec = self.vllm_config.speculative_config
            decode = 1 + (spec.num_speculative_tokens if spec is not None else 0)
            prefill = sum(n for n in out.num_scheduled_tokens.values() if n > decode)
            if prefill >= threshold:
                self._tpu_decode_only_left = window
        return out

    schedule._tpu_decode_priority = True
    Scheduler.schedule = schedule
    logger.info(
        "Decode priority: %d decode-only steps after a step with >= %d prefill tokens",
        window, threshold,
    )
