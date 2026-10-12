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
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput

import vllm_torchtpu.envs as envs

logger = init_logger(__name__)

# Cloud TPU v7x systolic array dimension alignment (#xla-chunked-prefill-tune).
SYSTOLIC_BUCKET_ALIGNMENT: int = 128


def align_to_systolic_bucket(token_cap: int) -> int:
    """Align a positive token budget cap up to a multiple of 128 tokens."""
    if token_cap <= 0:
        raise ValueError(
            f"token_cap must be positive when aligning to systolic bucket, "
            f"got {token_cap}. "
            "Set VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP to a positive "
            "multiple of 128 (e.g. 256) or 0/unset to disable."
        )
    return max(
        SYSTOLIC_BUCKET_ALIGNMENT,
        ((int(token_cap) + SYSTOLIC_BUCKET_ALIGNMENT - 1) // SYSTOLIC_BUCKET_ALIGNMENT)
        * SYSTOLIC_BUCKET_ALIGNMENT,
    )


def get_dp_decode_aware_prefill_chunk_cap() -> int | None:
    """Resolve and validate VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP.

    Returns None when unset or 0 (disabled). Raises ValueError on negative
    values. Rounds positive values up to a 128-token systolic multiple.
    """
    raw_cap = envs.VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP
    if (
        raw_cap is None
        and "VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP" in envs.__dict__
        and hasattr(envs, "__getattr__")
    ):
        raw_cap = envs.__getattr__("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP")
    if raw_cap is None:
        return None
    cap = int(raw_cap)
    if cap < 0:
        raise ValueError(
            f"VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP must be >= 0, got {cap}. "
            "Use a positive multiple of 128 (e.g. 256) or 0/unset to disable."
        )
    if cap == 0:
        return None
    return align_to_systolic_bucket(cap)


# Process-level cross-DP decode activity indicator updated by
# TPUModelRunner._dp_coordinated_step (RC37-B).
_PROCESS_CROSS_DP_HAS_ACTIVE_DECODE: bool = False


def set_process_cross_dp_active_decode(active: bool) -> None:
    """Update the process-level cross-DP active decode indicator."""
    global _PROCESS_CROSS_DP_HAS_ACTIVE_DECODE
    _PROCESS_CROSS_DP_HAS_ACTIVE_DECODE = bool(active)


def get_process_cross_dp_active_decode() -> bool:
    """Return the process-level cross-DP active decode indicator."""
    return bool(_PROCESS_CROSS_DP_HAS_ACTIVE_DECODE)


def is_active_decode_request(req: Any) -> bool:
    """Return True iff ``req`` is in the decode/spec-verify phase.

    A request in ``scheduler.running`` is in decode iff it has already computed
    all of its prompt tokens (``num_computed_tokens >= num_prompt_tokens``) and
    still has tokens to generate or verify. Partially-chunked prefill requests
    in ``running`` (``0 < num_computed_tokens < num_prompt_tokens`` or
    ``is_prefill_chunk=True``) return False so pure chunked-prefill steps are
    never misclassified as decode.

    Under ``AsyncScheduler`` (``--async-scheduling``), ``_update_after_schedule``
    adds ``1 + K`` to both ``num_computed_tokens`` and ``num_output_placeholders``
    at the end of Step 1, so on steady-state decode steps (Step 2..N)
    ``num_tokens_with_spec <= num_computed_tokens`` while
    ``num_tokens_with_spec + num_output_placeholders > num_computed_tokens``
    (matching ``Scheduler.schedule()`` at ``vllm/v1/core/sched/scheduler.py:493-524``).
    """
    if getattr(req, "is_prefill_chunk", False):
        return False
    num_computed = int(getattr(req, "num_computed_tokens", 0) or 0)
    if num_computed <= 0:
        return False
    num_prompt = getattr(req, "num_prompt_tokens", None)
    if num_prompt is not None and num_computed < int(num_prompt):
        return False
    num_placeholders = max(0, int(getattr(req, "num_output_placeholders", 0) or 0))
    if (
        num_placeholders > 0
        and num_prompt is not None
        and getattr(req, "max_tokens", None) is not None
        and num_computed + 2 - num_placeholders >= int(num_prompt) + int(req.max_tokens)
    ):
        return False
    num_tokens_with_spec = getattr(req, "num_tokens_with_spec", None)
    return not (
        num_tokens_with_spec is not None
        and (int(num_tokens_with_spec) + num_placeholders) <= num_computed
    )


def count_active_decode_requests_and_tokens(
    running: Iterable[Any],
) -> tuple[int, int]:
    """Count active decode requests and their scheduled token demand."""
    num_decode_reqs = 0
    decode_tokens = 0
    for req in running:
        if not is_active_decode_request(req):
            continue
        num_decode_reqs += 1
        num_computed = int(getattr(req, "num_computed_tokens", 0) or 0)
        num_placeholders = max(0, int(getattr(req, "num_output_placeholders", 0) or 0))
        num_with_spec = getattr(req, "num_tokens_with_spec", None)
        if (
            num_with_spec is not None
            and (int(num_with_spec) + num_placeholders) > num_computed
        ):
            req_tokens = (int(num_with_spec) + num_placeholders) - num_computed
        else:
            spec_tokens = getattr(req, "spec_token_ids", None)
            req_tokens = 1 + (len(spec_tokens) if spec_tokens is not None else 0)
        decode_tokens += max(1, req_tokens)
    return num_decode_reqs, decode_tokens


def compute_decode_aware_step_token_budget(
    max_num_scheduled_tokens: int,
    running: Iterable[Any],
    *,
    cross_dp_has_active_decode: bool = False,
    chunk_cap: int | None = None,
) -> tuple[int, bool, int, int]:
    """Compute the per-step token budget under decode-aware prefill clamping.

    Returns ``(effective_budget, clamped, num_decode_reqs, decode_tokens)``.
    When ``chunk_cap`` is active and either local ``running`` has active decode
    requests or ``cross_dp_has_active_decode`` is True, the step token budget is
    clamped to ``min(max_num_scheduled_tokens, max(aligned_cap, decode_tokens))``
    so any piggybacked prefill chunk stays within the micro-chunk bucket (e.g.
    256 tokens instead of 1024) while never starving active decode requests.
    Pure-prefill steps remain unclamped at ``max_num_scheduled_tokens``.
    """
    base_budget = int(max_num_scheduled_tokens)
    effective_cap = (
        align_to_systolic_bucket(chunk_cap)
        if chunk_cap is not None and chunk_cap > 0
        else get_dp_decode_aware_prefill_chunk_cap()
    )
    num_decode_reqs, decode_tokens = count_active_decode_requests_and_tokens(running)
    if effective_cap is None:
        return base_budget, False, num_decode_reqs, decode_tokens
    if num_decode_reqs == 0 and not cross_dp_has_active_decode:
        return base_budget, False, 0, 0
    target_budget = max(effective_cap, align_to_systolic_bucket(max(1, decode_tokens)))
    if target_budget >= base_budget:
        return base_budget, False, num_decode_reqs, decode_tokens
    return target_budget, True, num_decode_reqs, decode_tokens


class TpuDpScheduler(AsyncScheduler):
    """Scheduler that aligns prefill admission and micro-chunk budgets across DP ranks.

    vLLM passes a DP-synchronized scheduling signal through ``throttle_prefills``.
    This scheduler uses that signal to decide when to admit new prefill
    requests until all pending prefills are drained, and optionally clamps the
    per-step prefill chunk budget (``VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP``)
    whenever active decode requests are present so mixed prefill+decode steps
    execute in a small systolic token bucket (e.g. 256) instead of stalling
    decode TPOT at the full 1024-token prefill ceiling.
    """

    _cross_dp_has_active_decode: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drain_prefill = False
        self._cross_dp_has_active_decode = False
        self._cross_dp_synced_from_process = False
        set_process_cross_dp_active_decode(False)
        self._last_step_budget_info: dict[str, int | bool] = {
            "base_budget": int(getattr(self, "max_num_scheduled_tokens", 0) or 0),
            "effective_budget": int(getattr(self, "max_num_scheduled_tokens", 0) or 0),
            "clamped": False,
            "num_decode_reqs": 0,
            "decode_tokens": 0,
        }
        logger.info("TpuDpScheduler active")

    def set_cross_dp_active_decode(self, active: bool) -> None:
        """Update cross-DP decode activity indicator from coordinated DP step."""
        self._cross_dp_has_active_decode = bool(active)
        self._cross_dp_synced_from_process = bool(active)
        set_process_cross_dp_active_decode(bool(active))

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: Any,
    ) -> Any:
        if getattr(model_runner_output, "dp_step_has_active_decode", None) is not None:
            self.set_cross_dp_active_decode(
                bool(model_runner_output.dp_step_has_active_decode)
            )
        return super().update_from_output(scheduler_output, model_runner_output)

    def _schedule_paused_new(self) -> SchedulerOutput:
        """Run one step while keeping newly prefills buffered."""
        prev_pause_state = getattr(self, "_pause_state", PauseState.UNPAUSED)
        if prev_pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            return super().schedule(False)
        finally:
            self._pause_state = prev_pause_state

    def _resolve_base_token_budget(self) -> int:
        if hasattr(self, "max_num_scheduled_tokens") and self.max_num_scheduled_tokens:
            return int(self.max_num_scheduled_tokens)
        sched_cfg = getattr(self, "scheduler_config", None)
        if sched_cfg is not None and getattr(sched_cfg, "max_num_batched_tokens", None):
            return int(sched_cfg.max_num_batched_tokens)
        return 1024

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.prefill_capacity_bound = True
        if not hasattr(self, "_drain_prefill"):
            self._drain_prefill = False
        if not throttle_prefills:
            # Open window for draining accumulated prefills.
            self._drain_prefill = True
        if self._drain_prefill and not (self.waiting or self.skipped_waiting):
            # In drain mode until all accumulated prefills are scheduled.
            self._drain_prefill = False

        base_budget = self._resolve_base_token_budget()
        running_reqs = getattr(self, "running", ()) or ()
        proc_cross_dp = get_process_cross_dp_active_decode()
        if getattr(self, "_cross_dp_synced_from_process", False) and not proc_cross_dp:
            self._cross_dp_has_active_decode = False
            self._cross_dp_synced_from_process = False
        cross_dp_has_active_decode = bool(
            getattr(self, "_cross_dp_has_active_decode", False) or proc_cross_dp
        )
        if proc_cross_dp:
            self._cross_dp_synced_from_process = True
        self._cross_dp_has_active_decode = cross_dp_has_active_decode
        effective_budget, clamped, num_decode_reqs, decode_tokens = (
            compute_decode_aware_step_token_budget(
                base_budget,
                running_reqs,
                cross_dp_has_active_decode=cross_dp_has_active_decode,
            )
        )
        self._last_decode_aware_clamped = bool(clamped)
        if clamped:
            self._decode_aware_clamped_steps = (
                int(getattr(self, "_decode_aware_clamped_steps", 0) or 0) + 1
            )
        self._last_step_budget_info = {
            "base_budget": base_budget,
            "effective_budget": effective_budget,
            "clamped": clamped,
            "num_decode_reqs": num_decode_reqs,
            "decode_tokens": decode_tokens,
        }

        has_max_scheduled = hasattr(self, "max_num_scheduled_tokens")
        orig_max_scheduled = (
            self.max_num_scheduled_tokens if has_max_scheduled else None
        )
        sched_cfg = getattr(self, "scheduler_config", None)
        orig_cfg_batched = (
            getattr(sched_cfg, "max_num_batched_tokens", None)
            if sched_cfg is not None
            else None
        )
        orig_long_prefill = (
            getattr(sched_cfg, "long_prefill_token_threshold", None)
            if sched_cfg is not None
            else None
        )

        if clamped:
            if has_max_scheduled:
                self.max_num_scheduled_tokens = effective_budget
            if sched_cfg is not None and orig_cfg_batched is not None:
                sched_cfg.max_num_batched_tokens = effective_budget
            if (
                sched_cfg is not None
                and orig_long_prefill is not None
                and int(orig_long_prefill) > 0
            ):
                sched_cfg.long_prefill_token_threshold = min(
                    int(orig_long_prefill), effective_budget
                )

        try:
            if self._drain_prefill:
                return super().schedule(False)
            return self._schedule_paused_new()
        finally:
            if clamped:
                if has_max_scheduled:
                    self.max_num_scheduled_tokens = orig_max_scheduled
                if sched_cfg is not None and orig_cfg_batched is not None:
                    sched_cfg.max_num_batched_tokens = orig_cfg_batched
                if sched_cfg is not None and orig_long_prefill is not None:
                    sched_cfg.long_prefill_token_threshold = orig_long_prefill
