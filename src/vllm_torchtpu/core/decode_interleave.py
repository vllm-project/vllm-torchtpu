# SPDX-License-Identifier: Apache-2.0
"""Decode-only steps between prefill chunks (``TPU_SCHED_DECODE_INTERLEAVE_*``,
``TPU_SCHED_DECODE_FINISH_*``).

With chunked prefill and a long prompt in progress, vLLM gives every step to
a full prefill chunk (mnbt 4096, ~340-400 ms on GLM-5.2 TP16xDCP8) and the
running decodes ride along, one decode step per chunk. Two step-level
policies hold the prefill back for a few decode-only steps (~10-30 ms each)
while decodes are running:

* interleave (``TPU_SCHED_DECODE_INTERLEAVE_STEPS=k``): after a step that ran
  prefill tokens with at least one decode in it, the next ``k`` steps run
  decodes only. ``TPU_SCHED_DECODE_INTERLEAVE_ADAPTIVE=1`` scales ``k`` by
  the previous step's prefill tokens over the step budget (a short tail
  chunk earns one decode step, a full chunk ``k``).
* finish (``TPU_SCHED_DECODE_FINISH_TOKENS=R``): while a running decode is
  within ``R`` tokens of its ``max_tokens``, steps run decodes only, at most
  ``TPU_SCHED_DECODE_FINISH_MAX_STEPS`` (default 32) in a row before one step
  with prefill. A decode that is about to finish then finishes before the
  next chunk instead of paying for it. Only meaningful when ``max_tokens``
  is the real output length (``ignore_eos`` benchmarks, e.g. AgentX); with a
  large ``max_tokens`` it never fires.

A decode-only step reuses vLLM's own DP prefill-deferral path
(``schedule(throttle_prefills=True)`` with ``prefill_capacity_bound`` False):
running prefill chunks are skipped and no waiting request is admitted, every
decode is scheduled as usual. The step budget, the token buckets and the
decode shapes are unchanged, so no new compile shape appears. A step is only
deferred when at least one decode is schedulable, so a deferred step is never
empty. Wraps whatever ``Scheduler.schedule`` is installed (e.g. the SRPT
wrapper); on a deferred step the wrapper underneath is bypassed when it
exposes ``_tpu_orig`` (SRPT does), so its hold-back bookkeeping only sees
steps that can run prefill.

Off by default; with every knob at 0 the scheduler is not patched.
"""

from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LOG_EVERY = 5000


def _schedulable_decodes(scheduler: Any) -> list[Any]:
    """Running decodes that schedule() would give tokens this step (mirrors
    its async max_tokens skip and the PP decode cadence)."""
    out = []
    step = getattr(scheduler, "current_step", 0) + 1
    for r in scheduler.running:
        if r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens:
            continue
        if (
            r.num_output_placeholders > 0
            and r.num_computed_tokens + 2 - r.num_output_placeholders
            >= r.num_prompt_tokens + r.max_tokens
        ):
            continue
        if step < getattr(r, "next_decode_eligible_step", 0):
            continue
        out.append(r)
    return out


def _remaining_tokens(r: Any) -> int:
    return r.max_tokens - r.num_output_tokens - r.num_output_placeholders


class DecodeInterleavePolicy:
    def __init__(
        self,
        steps: int = 0,
        adaptive: bool = False,
        finish_tokens: int = 0,
        finish_max_steps: int = 32,
    ) -> None:
        for name, v in (
            ("TPU_SCHED_DECODE_INTERLEAVE_STEPS", steps),
            ("TPU_SCHED_DECODE_FINISH_TOKENS", finish_tokens),
        ):
            if v < 0:
                raise ValueError(f"{name} must not be negative: {v}")
        if finish_tokens and finish_max_steps < 1:
            raise ValueError(
                f"TPU_SCHED_DECODE_FINISH_MAX_STEPS must be >= 1: {finish_max_steps}"
            )
        self.steps = steps
        self.adaptive = adaptive
        self.finish_tokens = finish_tokens
        self.finish_max_steps = finish_max_steps
        self._left = 0  # interleave steps still owed
        self._finish_streak = 0
        self.n_steps = 0
        self.n_deferred = 0

    @property
    def enabled(self) -> bool:
        return bool(self.steps or self.finish_tokens)

    def want_defer(self, scheduler: Any) -> bool:
        decodes = _schedulable_decodes(scheduler)
        if not decodes:
            self._left = 0
            return False
        pending_prefill = bool(scheduler.waiting) or any(
            r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens
            for r in scheduler.running
        )
        if not pending_prefill:
            return False
        if self.steps and self._left > 0:
            return True
        return bool(
            self.finish_tokens
            and self._finish_streak < self.finish_max_steps
            and any(_remaining_tokens(r) <= self.finish_tokens for r in decodes)
        )

    def observe(self, scheduler: Any, out: Any, deferred: bool, decode_ids) -> None:
        scheduled = out.num_scheduled_tokens
        if not scheduled:
            return
        self.n_steps += 1
        if deferred:
            self.n_deferred += 1
            if self.steps and self._left > 0:
                self._left -= 1
            if self.finish_tokens:
                self._finish_streak += 1
        else:
            self._finish_streak = 0
            pf_tokens = sum(n for rid, n in scheduled.items() if rid not in decode_ids)
            n_dec = sum(1 for rid in scheduled if rid in decode_ids)
            if self.steps:
                if pf_tokens and n_dec:
                    if self.adaptive:
                        budget = max(scheduler.max_num_scheduled_tokens, 1)
                        self._left = -(-self.steps * pf_tokens // budget)
                    else:
                        self._left = self.steps
                else:
                    self._left = 0
        if self.n_steps % _LOG_EVERY == 0:
            logger.info(
                "Decode interleave: %d steps, %d decode-only (deferred prefill)",
                self.n_steps,
                self.n_deferred,
            )


def schedule_with_interleave(
    scheduler: Any, policy: DecodeInterleavePolicy, orig, throttle_prefills: bool
):
    defer = policy.want_defer(scheduler)
    decode_ids = {r.request_id for r in _schedulable_decodes(scheduler)}
    if not defer:
        out = orig(scheduler, throttle_prefills)
        policy.observe(scheduler, out, False, decode_ids)
        return out
    base = getattr(orig, "_tpu_orig", orig)
    saved = scheduler.prefill_capacity_bound
    scheduler.prefill_capacity_bound = False
    try:
        out = base(scheduler, True)
    finally:
        scheduler.prefill_capacity_bound = saved
    policy.observe(scheduler, out, True, decode_ids)
    return out


def patch_scheduler_decode_interleave() -> None:
    """Wrap ``Scheduler.schedule`` when a decode-interleave knob is set. Must
    run in the engine-core process, after the SRPT patch."""
    from vllm_torchtpu import envs

    kw = dict(
        steps=envs.TPU_SCHED_DECODE_INTERLEAVE_STEPS,
        adaptive=envs.TPU_SCHED_DECODE_INTERLEAVE_ADAPTIVE,
        finish_tokens=envs.TPU_SCHED_DECODE_FINISH_TOKENS,
        finish_max_steps=envs.TPU_SCHED_DECODE_FINISH_MAX_STEPS,
    )
    if not DecodeInterleavePolicy(**kw).enabled:  # also validates
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_tpu_decode_interleave", False):
        return
    orig = Scheduler.schedule

    def schedule(self, throttle_prefills: bool = False):
        policy = self.__dict__.get("_tpu_interleave_policy")
        if policy is None:
            policy = DecodeInterleavePolicy(**kw)
            self._tpu_interleave_policy = policy
            logger.info(
                "Decode interleave active on %s: interleave steps %d%s, finish "
                "tokens %d (max %d steps in a row), budget %d",
                type(self).__name__,
                kw["steps"],
                " (adaptive)" if kw["adaptive"] else "",
                kw["finish_tokens"],
                kw["finish_max_steps"],
                self.max_num_scheduled_tokens,
            )
        return schedule_with_interleave(self, policy, orig, throttle_prefills)

    Scheduler.schedule = schedule
    Scheduler._tpu_decode_interleave = True
