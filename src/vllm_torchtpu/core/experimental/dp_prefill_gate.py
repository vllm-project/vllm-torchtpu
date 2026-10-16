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
"""Scheduler that measures, and optionally tightens, vLLM's DP prefill hold-back.

``DPEngineCoreProc`` passes ``throttle_prefills=True`` on every step that is not
a multiple of ``--prefill-schedule-interval``, and ``Scheduler.schedule`` then
holds back prefill compute, except on a rank that:

* is ``prefill_capacity_bound``: its last step that was not held back left
  requests waiting, for any reason; or
* has no decoding request, so holding back would leave it nothing to run.

Under expert parallelism the DP ranks run every forward in lockstep and pad to
the largest rank's token count, so one rank that skips the hold-back costs
every rank a prefill-sized forward.

This scheduler counts, per rank, why held-back steps still carried prefill
tokens and why cadence-aligned steps left requests waiting, and logs the counts
every ``VLLM_TPU_PREFILL_GATE_LOG_STEPS`` steps. With
``VLLM_TPU_PREFILL_GATE_MIN_CADENCES=N`` (N >= 1) it also replaces upstream's
capacity-bound test: a rank counts as capacity-bound only after N consecutive
cadence-aligned steps that exhausted the token budget and still left
admissible requests waiting. That is the only case in which prefilling more
often lets the rank admit more work; a queue held back by ``max_num_seqs`` or
KV capacity does not drain faster. N=0 leaves upstream's flag untouched.

With ``--prefill-schedule-interval=1`` no step is held back, so the scheduler
only collects stats. It extends :class:`SrptPrefillScheduler` (and through it
:class:`PrefixAwareScheduler`), so it keeps decode-first / shortest-remaining
RUNNING order, SRPT hold-back of long running chunks on cadence-aligned steps,
and warmest-prefix-first admission; it expects ``--async-scheduling``. SRPT
never holds on a throttled step, so there upstream's DP hold-back decides and
this class only rewrites ``prefill_capacity_bound``. Chunks SRPT holds are
restored before this class reads the step, and are not counted as prefill.

Select with ``--scheduler-cls``
``vllm_torchtpu.core.experimental.dp_prefill_gate.DpPrefillGateScheduler``.
"""

from collections import Counter
from itertools import chain

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu import envs
from vllm_torchtpu.core.experimental.srpt_prefill_scheduler import SrptPrefillScheduler
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class DpPrefillGateScheduler(SrptPrefillScheduler):
    """Attribute, and optionally gate, DP prefill hold-back escapes per rank."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_min_cadences = envs.VLLM_TPU_PREFILL_GATE_MIN_CADENCES
        self._gate_log_steps = envs.VLLM_TPU_PREFILL_GATE_LOG_STEPS
        self._gate_streak = 0
        self._gate_stats: Counter[str] = Counter()
        self._gate_dp_rank = self.parallel_config.data_parallel_index
        interval = self.scheduler_config.prefill_schedule_interval
        dp_size = self.parallel_config.data_parallel_size
        logger.info(
            "DpPrefillGateScheduler active (dp_rank=%d, dp_size=%d, "
            "prefill_schedule_interval=%d, min_cadences=%d, log_steps=%d)",
            self._gate_dp_rank,
            dp_size,
            interval,
            self._gate_min_cadences,
            self._gate_log_steps,
        )
        if self._gate_min_cadences and (interval <= 1 or dp_size <= 1):
            logger.warning(
                "DpPrefillGateScheduler: VLLM_TPU_PREFILL_GATE_MIN_CADENCES=%d "
                "has no effect with --prefill-schedule-interval=%d and "
                "--data-parallel-size=%d: prefills are only held back with "
                "both > 1.",
                self._gate_min_cadences,
                interval,
                dp_size,
            )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if not self._gate_min_cadences and not self._gate_log_steps:
            return super().schedule(throttle_prefills)

        # Upstream decides whether to hold back from these; read them before
        # super() rewrites prefill_capacity_bound and is_prefill_chunk.
        bound_before = self.prefill_capacity_bound
        prefilling_before = {r.request_id: r.is_prefill_chunk for r in self.running}
        has_decoder = not all(prefilling_before.values())

        output = super().schedule(throttle_prefills)

        prefill_tokens = self._prefill_tokens(output, prefilling_before)
        waiting = self._has_admissible_waiting()
        budget_full = self._budget_exhausted(output)
        seqs_full = self._active_reqs_full()

        if self._gate_min_cadences:
            starved = waiting and budget_full
            if not throttle_prefills:
                self._gate_streak = self._gate_streak + 1 if starved else 0
            elif bound_before and not starved:
                # An escape step that no longer exhausts the budget (queue
                # drained, or now held back by seqs/KV): stop escaping.
                self._gate_streak = 0
            self.prefill_capacity_bound = self._gate_streak >= self._gate_min_cadences

        if self._gate_log_steps:
            self._record(
                throttle_prefills,
                bound_before,
                has_decoder,
                prefill_tokens,
                waiting,
                budget_full,
                seqs_full,
            )
        return output

    @staticmethod
    def _prefill_tokens(
        output: SchedulerOutput, prefilling_before: dict[str, bool]
    ) -> int:
        """Tokens the step scheduled for prompt computation.

        A running request counts if it was a prefill chunk before the step, so
        a speculative verify (``1 + K`` tokens) does not. A request admitted
        from a waiting queue counts unless it only needed its last prompt
        token, which upstream schedules as a (spec-padded) decode even on a
        held-back step.
        """
        spec_tokens = output.scheduled_spec_decode_tokens
        tokens = 0
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            prefilling = prefilling_before.get(req_id)
            if prefilling is None:
                prefilling = num_tokens - len(spec_tokens.get(req_id, ())) > 1
            if prefilling:
                tokens += num_tokens
        return tokens

    def _has_admissible_waiting(self) -> bool:
        """Whether a request the waiting scan could admit is still queued.

        Requests the scan deferred this step and requests in a blocked status
        (remote KV, grammar, streaming input) do not count, as in upstream's
        capacity-bound test.
        """
        # vLLM < 0.31 has neither queue: it parks skipped and blocked requests
        # in skipped_waiting, which upstream's test ignores as well.
        kv_holding = getattr(self, "kv_holding_waiting", ())
        deferred = getattr(self, "deferred_waiting", ())
        if len(self.waiting) + len(kv_holding) <= len(deferred):
            return False
        return any(
            r not in deferred and not self._is_blocked_waiting_status(r.status)
            for r in chain(kv_holding, self.waiting)
        )

    def _active_reqs_full(self) -> bool:
        """Whether the waiting scan stopped at the active-request limit."""
        # Paused streaming sessions hold a request slot outside `running`.
        max_active = getattr(self, "max_num_active_reqs", self.max_num_running_reqs)
        num_active = len(self.running) + self.num_waiting_for_streaming_input
        return num_active >= max_active

    def _budget_exhausted(self, output: SchedulerOutput) -> bool:
        """Whether the step left no room for another request's tokens.

        Mirrors upstream's two budgets: ``max_num_scheduled_tokens``, and
        ``max_num_batched_tokens``, which also reserves ``draft_slots`` per
        scheduled request under speculative decoding. A remainder of at most
        ``num_spec_tokens`` counts as exhausted: upstream will not admit a
        decode it cannot pad to ``1 + num_spec_tokens``.
        """
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        total = output.total_num_scheduled_tokens
        token_left = self.max_num_scheduled_tokens - total
        input_left = (
            self.scheduler_config.max_num_batched_tokens
            - total
            - len(output.num_scheduled_tokens) * draft_slots
        )
        return min(token_left, input_left - draft_slots) <= self.num_spec_tokens

    def _record(
        self,
        throttled: bool,
        bound_before: bool,
        has_decoder: bool,
        prefill_tokens: int,
        waiting: bool,
        budget_full: bool,
        seqs_full: bool,
    ) -> None:
        s = self._gate_stats
        s["steps"] += 1
        s["bound_steps"] += self.prefill_capacity_bound
        if throttled:
            s["throttled"] += 1
            if prefill_tokens:
                if bound_before:
                    reason = "bound"
                elif not has_decoder:
                    reason = "no_decoder"
                else:
                    reason = "other"
                s[f"throttled_prefill_{reason}"] += 1
                s["throttled_prefill_tokens"] += prefill_tokens
        else:
            s["cadence"] += 1
            if prefill_tokens:
                s["cadence_prefill"] += 1
                s["cadence_prefill_tokens"] += prefill_tokens
            if waiting:
                if budget_full:
                    s["cadence_waiting_budget"] += 1
                elif seqs_full:
                    s["cadence_waiting_seqs"] += 1
                else:
                    s["cadence_waiting_other"] += 1
        if s["steps"] >= self._gate_log_steps:
            logger.info(
                "DP prefill gate: dp_rank=%d min_cadences=%d steps=%d "
                "throttled=%d throttled_prefill[bound=%d no_decoder=%d "
                "other=%d tokens=%d] cadence=%d cadence_prefill=%d tokens=%d "
                "cadence_waiting[budget=%d seqs=%d other=%d] bound_steps=%d",
                self._gate_dp_rank,
                self._gate_min_cadences,
                s["steps"],
                s["throttled"],
                s["throttled_prefill_bound"],
                s["throttled_prefill_no_decoder"],
                s["throttled_prefill_other"],
                s["throttled_prefill_tokens"],
                s["cadence"],
                s["cadence_prefill"],
                s["cadence_prefill_tokens"],
                s["cadence_waiting_budget"],
                s["cadence_waiting_seqs"],
                s["cadence_waiting_other"],
                s["bound_steps"],
            )
            s.clear()
