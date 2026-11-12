# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded hold of a prefill's Mamba-aligned tail (VLLM_TPU_TAIL_HOLD_STEPS).

In Mamba "align" mode the scheduler ends a prompt's last full chunk at its
last block boundary, so the boundary state is written at the step end, and
runs the rest of the prompt (at most one block) as a step of its own. When
nothing else prefills in that step, it pays the full fixed cost of a prefill
step for a few hundred rows.

With ``VLLM_TPU_TAIL_HOLD_STEPS=K`` the scheduler holds such a tail for up to
K steps, but only while

- no other prefill can run in this step (no other running request has
  prefill left, and no waiting request can be admitted), and
- a waiting request is queued for a seat (a partner prefill is coming), and
- some other running request still runs in the step (the step is not empty).

The tail is released as soon as it can share a step with another prefill, or
when any held tail has waited K steps (then every held tail runs in that
step). A held request keeps its blocks and state; when it runs, the upstream
split sees exactly the same request state, so its chunks stay
``[s, B_last)`` + ``[B_last, isl)`` and the Mamba checkpoint, the prefix-cache
hashes and the DCP page ownership are unchanged. Only the step the tail runs
in changes.

The hold is applied through ``_mamba_block_aligned_split`` returning 0 for a
held request: the running loop already skips a request with 0 new tokens.
"""

import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LOG_INTERVAL_S = 60.0


def is_mamba_tail(request, align: int, max_tail: int) -> bool:
    """Whether ``request`` (running) has only a short prompt remainder left
    (at most ``max_tail`` tokens), starting on a multiple of ``align``: the
    piece the align split runs as a step of its own. Resumed requests
    replaying output tokens are left out. Holding any such request is safe:
    the split is recomputed from the same state when it runs."""
    if getattr(request, "num_output_tokens", 0) > 0:
        return False
    computed = request.num_computed_tokens
    prompt = request.num_prompt_tokens
    if computed <= 0 or computed >= prompt or computed % align:
        return False
    return prompt - computed <= max_tail


def tail_geometry(scheduler) -> tuple[int, int]:
    """(align, max_tail) for ``is_mamba_tail``. The split runs on the
    physical cache block or, under the TPU scheduler-block patch, on the
    scheduler block; take the finer one for the boundary and the coarser one
    (two with the EAGLE block drop's back-off) for the tail length."""
    sizes = [
        int(b)
        for b in (
            scheduler.cache_config.block_size,
            getattr(scheduler, "block_size", None),
        )
        if b
    ]
    eagle_drop = bool(getattr(scheduler, "use_eagle_block_drop", False))
    return min(sizes), max(sizes) * (2 if eagle_drop else 1)


class TailHold:
    """Per-scheduler hold state. ``plan`` runs once per ``schedule()``, before
    the upstream schedule; ``held_now`` is the set of request ids to skip."""

    def __init__(self, steps: int) -> None:
        self.steps = int(steps)
        self.waited: dict[str, int] = {}
        self.held_now: frozenset[str] = frozenset()
        # Counters (tail-steps held; tails released with a partner prefill,
        # after waiting; tails run after K steps).
        self.n_hold_steps = 0
        self.n_merged = 0
        self.n_expired = 0
        self._last_log = 0.0
        self._logged_on = False

    def plan(self, scheduler, *, seat_free: bool) -> frozenset[str]:
        self.held_now = frozenset()
        if self.steps <= 0 or not getattr(
            scheduler, "need_mamba_block_aligned_split", False
        ):
            return self.held_now
        if not self._logged_on:
            self._logged_on = True
            logger.info(
                "VLLM_TPU_TAIL_HOLD_STEPS=%d: a Mamba-aligned prefill tail waits "
                "up to %d steps for another prefill to share its step.",
                self.steps,
                self.steps,
            )
        align, max_tail = tail_geometry(scheduler)
        running = scheduler.running
        tails = [r for r in running if is_mamba_tail(r, align, max_tail)]
        tail_ids = {r.request_id for r in tails}
        # Forget requests that ran their tail, finished or were preempted.
        for rid in [rid for rid in self.waited if rid not in tail_ids]:
            del self.waited[rid]
        if not tails:
            return self.held_now

        other_prefill = any(
            r.is_prefill_chunk and r.request_id not in tail_ids for r in running
        )
        num_waiting = scheduler.get_request_counts()[1]
        partner_now = other_prefill or (num_waiting > 0 and seat_free)
        partner_pending = num_waiting > 0
        step_not_empty = len(running) > len(tails)

        if partner_now or not partner_pending or not step_not_empty:
            # Runs now, as upstream: with a partner, or alone because no
            # partner can come (never starve a tail on an idle queue).
            if partner_now:
                self.n_merged += sum(1 for rid in tail_ids if self.waited.get(rid))
            self._maybe_log()
            return self.held_now
        if any(self.waited.get(rid, 0) >= self.steps for rid in tail_ids):
            # A tail used up its K steps: it runs alone now, and every other
            # held tail shares its step.
            self.n_expired += sum(
                1 for rid in tail_ids if self.waited.get(rid, 0) >= self.steps
            )
            self.n_merged += sum(
                1 for rid in tail_ids if 0 < self.waited.get(rid, 0) < self.steps
            )
            self._maybe_log()
            return self.held_now
        for rid in tail_ids:
            self.waited[rid] = self.waited.get(rid, 0) + 1
        self.n_hold_steps += len(tail_ids)
        self.held_now = frozenset(tail_ids)
        self._maybe_log()
        return self.held_now

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < _LOG_INTERVAL_S:
            return
        if not (self.n_hold_steps or self.n_merged or self.n_expired):
            return
        self._last_log = now
        logger.info(
            "VLLM_TPU_TAIL_HOLD: held %d tail-steps; merged %d tails into a "
            "prefill step; %d ran alone after %d steps",
            self.n_hold_steps,
            self.n_merged,
            self.n_expired,
            self.steps,
        )
