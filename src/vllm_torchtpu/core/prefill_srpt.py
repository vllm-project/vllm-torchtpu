# SPDX-License-Identifier: Apache-2.0
"""Shortest-remaining-prefill-first with hold-back (``TPU_SCHED_PREFILL_SRPT``).

vLLM's ``schedule()`` serves ``running`` before ``waiting``. With one
in-progress chunked prefill holding the whole token budget (mnbt 4096, no
long-prefill threshold), a short prompt that arrives while a long prompt is
being chunked waits for every remaining chunk of the long one: head-of-line
blocking, the largest component of the AgentX P90 TTFT tail.

This policy orders prefill work shortest-remaining-first across in-progress
prefills AND waiting requests. Remaining tokens are ``num_tokens -
num_computed_tokens`` for a running prefill, and the prompt tokens not already
in the prefix cache for a waiting request (the same lookup the
PrefixAwareScheduler uses, which records no stats). Per step:

* a running prefill is *held back* (gets no tokens this step) when an
  admissible waiting request, or a running prefill behind it in ``running``,
  has fewer remaining tokens;
* waiting requests with more remaining tokens than the shortest held prefill
  are hidden for the step, so the freed budget is not handed to a request
  longer than the one held back;
* the visible waiting requests are offered shortest first.

Decodes keep going first: schedule() sees ``running`` as the decodes, then
the prefills not held, each group in its running order (a request admitted
during a hold sits behind the held prefill in ``running`` and would otherwise
be starved by its chunks once it decodes). Held prefills are detached from
that view and the slot limit is lowered by the same count (a held request
keeps its slot); hidden requests are taken out of ``waiting``. Both lists are
put back in their original order right after the call, with vLLM's new
admissions after every original running request and its preemptions (always
the view's last entry) at the front of ``waiting``, so ``running[-1]`` is still
the newest admission. The policy changes only which requests get tokens, never
a step's token budget, so no new token padding or compile shape appears.

Off by default. No request is held when the scheduler is not FCFS, when a
preempted, resumed or non-WAITING request is queued, or in the step after a
held step that scheduled none of the prefills it favoured (no livelock when
the short request cannot be allocated). ``TPU_SCHED_PREFILL_SRPT_MAX_HOLD_STEPS
=K`` (0 = off) stops holding a prefill after K consecutive held steps. The
waiting key is read before schedule() runs, so blocks the previous step wrote
may still count as uncached under the same-step prefix-hit patch.
"""

from dataclasses import dataclass, field
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LOG_EVERY = 2000
_KEY_TTL_STEPS = 8


@dataclass
class SrptPlan:
    # Running prefills that get no tokens this step.
    held: list[Any] = field(default_factory=list)
    # Waiting requests taken out of the queue this step.
    hidden: list[Any] = field(default_factory=list)
    # ``running`` as schedule() sees it (None: unchanged): decodes first, then
    # the prefills not held, in their running order.
    view: list[Any] | None = None
    # ``waiting`` as schedule() sees it (None: unchanged), shortest first.
    waiting_view: list[Any] | None = None
    # Prefill ids the plan favoured (visible running prefills + visible waiting).
    favoured: set[str] = field(default_factory=set)

    @property
    def active(self) -> bool:
        return self.view is not None or self.waiting_view is not None


def _is_running_prefill(request: Any) -> bool:
    # A decode has num_computed_tokens == num_tokens - 1 (sync) or carries
    # placeholders (async); only an unfinished prompt counts.
    return bool(request.is_prefill_chunk) or (
        request.num_computed_tokens < request.num_prompt_tokens
    )


class PrefillSrptPolicy:
    def __init__(self, max_hold_steps: int = 0) -> None:
        if max_hold_steps < 0:
            raise ValueError(
                "TPU_SCHED_PREFILL_SRPT_MAX_HOLD_STEPS must not be negative: "
                f"{max_hold_steps}"
            )
        self.max_hold_steps = max_hold_steps
        self._skip_next = False
        self._held_streak: dict[str, int] = {}
        self._warmth_broken = False
        self._key_cache: dict[str, tuple[int, tuple[int, float]]] = {}
        self.steps = 0
        self.hold_steps = 0
        self.held_total = 0

    # -- keys ---------------------------------------------------------------
    def _warmth(self, scheduler: Any, request: Any) -> int:
        kvm = scheduler.kv_cache_manager
        if not getattr(kvm, "enable_caching", False) or getattr(
            request, "skip_reading_prefix_cache", False
        ):
            return 0
        if self._warmth_broken:
            return 0
        try:
            hit = kvm.coordinator.find_longest_cache_hit(
                request.block_hashes, max(request.num_tokens - 1, 0)
            )
            return int(hit[1])
        except Exception:
            self._warmth_broken = True
            logger.exception(
                "Prefill SRPT: prefix-cache lookup failed; waiting requests are "
                "keyed by their full prompt from now on."
            )
            return 0

    def _waiting_key(self, scheduler: Any, request: Any) -> tuple[int, float]:
        # The prefix lookup walks the prompt's block hashes; reuse it for a
        # few steps while the request waits.
        step = getattr(scheduler, "current_step", 0)
        cached = self._key_cache.get(request.request_id)
        if cached is not None and step - cached[0] < _KEY_TTL_STEPS:
            return cached[1]
        key = (
            request.num_tokens - self._warmth(scheduler, request),
            request.arrival_time,
        )
        self._key_cache[request.request_id] = (step, key)
        return key

    @staticmethod
    def _running_key(request: Any) -> tuple[int, float]:
        return (request.num_tokens - request.num_computed_tokens, request.arrival_time)

    # -- plan ---------------------------------------------------------------
    def _hold_eligible(self, scheduler: Any, waiting: list[Any]) -> bool:
        from vllm.v1.core.sched.request_queue import SchedulingPolicy
        from vllm.v1.request import RequestStatus

        if self._skip_next or scheduler.policy != SchedulingPolicy.FCFS:
            return False
        if getattr(scheduler, "skipped_waiting", None):
            return False
        return all(
            r.status == RequestStatus.WAITING
            and r.num_preemptions == 0
            and r.num_computed_tokens == 0
            for r in waiting
        )

    def plan(self, scheduler: Any) -> SrptPlan:
        plan = SrptPlan()
        running = scheduler.running
        waiting = list(scheduler.waiting)
        prefills = [r for r in running if _is_running_prefill(r)]
        decodes = [r for r in running if not _is_running_prefill(r)]
        if not self._hold_eligible(scheduler, waiting):
            # Decodes still go first: a request admitted during a hold sits
            # behind the held prefill in ``running``.
            view = decodes + prefills
            if any(a is not b for a, b in zip(view, running)):
                plan.view = view
            return plan
        if len(self._key_cache) > 4 * len(waiting) + 64:
            live = {r.request_id for r in waiting}
            self._key_cache = {k: v for k, v in self._key_cache.items() if k in live}

        num_running = len(running) + getattr(
            scheduler, "num_waiting_for_streaming_input", 0
        )
        slots_free = num_running < scheduler.max_num_running_reqs
        wkeys = (
            {r.request_id: self._waiting_key(scheduler, r) for r in waiting}
            if slots_free and (prefills or len(waiting) > 1)
            else {}
        )
        min_wait = min(wkeys.values()) if wkeys else None

        rkeys = [self._running_key(r) for r in prefills]
        for i, request in enumerate(prefills):
            key = rkeys[i]
            later = rkeys[i + 1 :]
            if (min_wait is not None and min_wait < key) or (
                later and min(later) < key
            ):
                streak = self._held_streak.get(request.request_id, 0)
                if self.max_hold_steps and streak >= self.max_hold_steps:
                    continue
                plan.held.append(request)
        held_ids = {id(r) for r in plan.held}
        view = decodes + [r for r in prefills if id(r) not in held_ids]
        if plan.held or any(a is not b for a, b in zip(view, running)):
            plan.view = view
        visible: list[Any] = []
        if wkeys:
            min_held = min((self._running_key(r) for r in plan.held), default=None)
            visible = [
                r
                for r in waiting
                if min_held is None or wkeys[r.request_id] <= min_held
            ]
            plan.hidden = [r for r in waiting if r not in visible]
            visible.sort(key=lambda r: wkeys[r.request_id])
            if plan.hidden or any(a is not b for a, b in zip(visible, waiting)):
                plan.waiting_view = visible
        plan.favoured = {r.request_id for r in prefills if id(r) not in held_ids} | {
            r.request_id for r in visible
        }
        return plan

    def observe(self, plan: SrptPlan, scheduler_output: Any) -> None:
        scheduled = scheduler_output.num_scheduled_tokens
        self._skip_next = False
        if plan.held:
            if not any(rid in scheduled for rid in plan.favoured):
                # The favoured prefills could not run (no KV, no budget):
                # leave the next step to vLLM so the held one makes progress.
                self._skip_next = True
            for request in plan.held:
                rid = request.request_id
                self._held_streak[rid] = self._held_streak.get(rid, 0) + 1
        held_ids = {r.request_id for r in plan.held}
        for rid in list(self._held_streak):
            if rid not in held_ids:
                del self._held_streak[rid]
        if not scheduled:
            return
        self.steps += 1
        self.hold_steps += bool(plan.held)
        self.held_total += len(plan.held)
        if self.steps % _LOG_EVERY == 0:
            logger.info(
                "Prefill SRPT: %d steps, %d with a prefill held back (%d holds)",
                self.steps,
                self.hold_steps,
                self.held_total,
            )


def _restore_order(original: list[Any], current: list[Any], extra: list[Any]):
    """Members of ``original`` still in ``current`` or ``extra``, in their
    original order, and the members ``current`` gained, in its order."""
    keep = {id(r) for r in current} | {id(r) for r in extra}
    orig_ids = {id(r) for r in original}
    survivors = [r for r in original if id(r) in keep]
    gained = [r for r in current if id(r) not in orig_ids]
    return survivors, gained


def schedule_with_srpt(
    scheduler: Any, policy: PrefillSrptPolicy, orig, throttle_prefills: bool
):
    """Run ``orig`` (vLLM's ``schedule``) for one step under ``policy``."""
    plan = policy.plan(scheduler)
    if not plan.active:
        out = orig(scheduler, throttle_prefills)
        policy.observe(plan, out)
        return out

    running = scheduler.running
    waiting = scheduler.waiting
    orig_running = list(running)
    orig_waiting = list(waiting)
    saved_slots = scheduler.max_num_running_reqs
    saved_bound = scheduler.prefill_capacity_bound

    if plan.view is not None:
        running[:] = plan.view
    if plan.waiting_view is not None:
        waiting.clear()
        waiting.extend(plan.waiting_view)
    # A held request keeps its slot.
    scheduler.max_num_running_reqs = saved_slots - len(plan.held)
    try:
        out = orig(scheduler, throttle_prefills)
    finally:
        scheduler.max_num_running_reqs = saved_slots
        # running: original order (held back in place), then new admissions.
        survivors, admitted = _restore_order(orig_running, list(running), plan.held)
        running[:] = survivors + admitted
        # waiting: original FCFS order; requests vLLM preempted this step were
        # prepended, admitted ones are gone.
        survivors_w, preempted = _restore_order(
            orig_waiting, list(waiting), plan.hidden
        )
        waiting.clear()
        waiting.extend(preempted + survivors_w)
        if plan.hidden and not throttle_prefills:
            # schedule() computes it from the queue it saw; count the hidden.
            scheduler.prefill_capacity_bound = saved_bound or bool(waiting)
    policy.observe(plan, out)
    return out


def patch_scheduler_prefill_srpt() -> None:
    """Wrap ``Scheduler.schedule`` when ``TPU_SCHED_PREFILL_SRPT`` is set. Must
    run in the engine-core process; subclasses (async, prefix-aware) inherit
    it."""
    from vllm_torchtpu import envs

    if not envs.TPU_SCHED_PREFILL_SRPT:
        return
    max_hold = envs.TPU_SCHED_PREFILL_SRPT_MAX_HOLD_STEPS
    PrefillSrptPolicy(max_hold)  # validates
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_tpu_prefill_srpt", False):
        return
    orig = Scheduler.schedule

    def schedule(self, throttle_prefills: bool = False):
        policy = self.__dict__.get("_tpu_srpt_policy")
        if policy is None:
            policy = PrefillSrptPolicy(max_hold)
            self._tpu_srpt_policy = policy
            logger.info(
                "Prefill SRPT with hold-back active on %s (budget %d, max hold "
                "steps %s)",
                type(self).__name__,
                self.max_num_scheduled_tokens,
                max_hold or "off",
            )
        return schedule_with_srpt(self, policy, orig, throttle_prefills)

    # The decode-interleave wrapper calls the unwrapped schedule on a
    # decode-only step, so a deferred step never counts as a hold.
    schedule._tpu_orig = orig  # pyrefly: ignore[missing-attribute]
    Scheduler.schedule = schedule
    Scheduler._tpu_prefill_srpt = True
