# SPDX-License-Identifier: Apache-2.0
"""Decode priority after a large prefill step (VLLM_TPU_DECODE_PRIORITY).

At c32 the p90-interactivity requests spend ~57% of their decode window inside
mixed steps carrying an 8K prefill chunk (j-aff13101 / j-63f072c8), and the
scheduler follows such a step with about one decode step before the next chunk.
With this flag, after any step that schedules >= N prefill tokens the next K
steps defer prefill work (in-progress chunks and new admissions) and run the
running decodes only, using vLLM's own DP prefill-balancing path
(`schedule(throttle_prefills=True)`). Prefill chunk sizes are unchanged; only
their spacing is.

Format: ``N:K`` or ``N:K:opt=val:...`` with options

* ``scale=S``: the window is ``ceil(K * prefill_tokens / S)`` steps (at least
  1), so a decode request keeps a roughly fixed share of device time whatever
  the chunk size. Without it the window is K steps.
* ``protect=M``: also defer prefill while some running decode request has at
  most M tokens left before its ``max_tokens``, for at most ``hold``
  consecutive steps (default 8). A short request then finishes before the next
  chunk lands in its decode window.
* ``minb=B``: open a window only when at least B requests are decoding. An
  inserted decode-only step at a small batch costs nearly a full step for few
  tokens; the p90 tail decodes in crowded periods, where the step is cheap per
  token.
* ``deadline=D``: never defer while a waiting or partially prefilled request
  has waited more than D ms since arrival (TTFT guard).

A step's prefill tokens are its requests' scheduled tokens above the decode
size (1 + speculative tokens). Deferral is skipped when nothing is decoding
(vLLM's `defer_prefills` then admits prefills anyway).
"""

import dataclasses
import functools
import math
import os
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class _Config:
    threshold: int
    window: int
    scale: int = 0
    protect: int = 0
    hold: int = 8
    deadline_ms: float = 0.0
    min_decodes: int = 0

    def window_for(self, prefill: int) -> int:
        if self.scale <= 0:
            return self.window
        return max(1, math.ceil(self.window * prefill / self.scale))


def _parse() -> _Config | None:
    raw = os.environ.get("VLLM_TPU_DECODE_PRIORITY", "").strip()
    if not raw:
        return None
    parts = raw.split(":")
    try:
        n, k = int(parts[0]), int(parts[1])
        opts = dict(p.split("=", 1) for p in parts[2:])
        cfg = _Config(
            n, k,
            scale=int(opts.pop("scale", 0)),
            protect=int(opts.pop("protect", 0)),
            hold=int(opts.pop("hold", 8)),
            deadline_ms=float(opts.pop("deadline", 0)),
            min_decodes=int(opts.pop("minb", 0)),
        )
    except (IndexError, ValueError) as e:
        raise ValueError(f"VLLM_TPU_DECODE_PRIORITY={raw!r}: want N:K[:opt=val...]") from e
    if n <= 0 or k <= 0 or opts or cfg.hold <= 0:
        raise ValueError(f"VLLM_TPU_DECODE_PRIORITY={raw!r}: want N:K with N, K > 0 and "
                         "options scale/protect/hold/deadline/minb")
    return cfg


def _is_decoding(request) -> bool:
    return request.num_computed_tokens >= request.num_prompt_tokens


def _near_done(sched, limit: int) -> bool:
    for r in sched.running:
        if _is_decoding(r) and r.max_tokens - r.num_output_tokens <= limit:
            return True
    return False


def _has_prefill_work(sched) -> bool:
    return len(sched.waiting) > 0 or any(not _is_decoding(r) for r in sched.running)


def _oldest_wait_ms(sched) -> float:
    arrivals = [r.arrival_time for r in sched.waiting]
    arrivals += [r.arrival_time for r in sched.running if not _is_decoding(r)]
    return (time.time() - min(arrivals)) * 1000.0 if arrivals else 0.0


def patch_decode_priority() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    cfg = _parse()
    if cfg is None or getattr(Scheduler.schedule, "_tpu_decode_priority", False):
        return
    original = Scheduler.schedule

    @functools.wraps(original)
    def schedule(self, throttle_prefills: bool = False):
        left = getattr(self, "_tpu_decode_only_left", 0)
        defer = False
        if left > 0:
            self._tpu_decode_only_left = left - 1
            defer = True
        elif cfg.protect and _has_prefill_work(self) and _near_done(self, cfg.protect):
            held = getattr(self, "_tpu_protect_held", 0)
            if held < cfg.hold:
                self._tpu_protect_held = held + 1
                defer = True
        else:
            self._tpu_protect_held = 0
        if defer and cfg.deadline_ms and _oldest_wait_ms(self) > cfg.deadline_ms:
            defer = False
            self._tpu_decode_only_left = 0
        if defer:
            throttle_prefills = True
            # vLLM releases deferral when the waiting queue outgrows the deferred
            # set (DP capacity rule); here the window is the whole point.
            self.prefill_capacity_bound = False
        out = original(self, throttle_prefills)
        if left <= 0:
            spec = self.vllm_config.speculative_config
            decode = 1 + (spec.num_speculative_tokens if spec is not None else 0)
            prefill = sum(n for n in out.num_scheduled_tokens.values() if n > decode)
            if prefill > 0:
                self._tpu_protect_held = 0  # a chunk ran: the next near-done request may hold again
            if prefill >= cfg.threshold and (
                not cfg.min_decodes
                or sum(1 for r in self.running if _is_decoding(r)) >= cfg.min_decodes
            ):
                self._tpu_decode_only_left = cfg.window_for(prefill)
                n = getattr(self, "_tpu_decode_windows", 0) + 1
                self._tpu_decode_windows = n
                if n in (1, 1000, 10000):  # path proof in the engine-core process
                    logger.info(
                        "Decode priority: window %d opened after %d prefill tokens (%d steps)",
                        n, prefill, self._tpu_decode_only_left,
                    )
        return out

    schedule._tpu_decode_priority = True
    Scheduler.schedule = schedule
    logger.info("Decode priority: %s", cfg)
