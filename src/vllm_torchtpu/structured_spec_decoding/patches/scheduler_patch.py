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
"""Scheduler patches for structured speculative decoding (GaaD)."""

import logging
from typing import Any

from vllm_torchtpu.structured_spec_decoding.core.draft_planner import (
    CompositeDraftPlanner,
)
from vllm_torchtpu.structured_spec_decoding.core.fsm_bridge import JumpStats

logger = logging.getLogger(__name__)
_STATS = JumpStats()


def get_stats() -> JumpStats:
    """Returns the global JumpStats instance."""
    return _STATS


def topup_scheduled_spec_decode_tokens(
    scheduler: Any,
    scheduler_output: Any,
) -> None:
    """Tops up validated async speculative draft tokens in scheduler_output."""
    if scheduler_output is None:
        return
    sched_spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
    if not sched_spec:
        return
    manager = getattr(scheduler, "structured_output_manager", None)
    tokenizer = getattr(manager, "tokenizer", None) if manager else None

    if getattr(scheduler_output, "scheduled_spec_decode_flags", None) is None:
        scheduler_output.scheduled_spec_decode_flags = {}
    sched_flags = scheduler_output.scheduled_spec_decode_flags
    num_invalid_map = getattr(scheduler_output, "num_invalid_spec_tokens", None)
    if num_invalid_map is None:
        num_invalid_map = {}
        scheduler_output.num_invalid_spec_tokens = num_invalid_map

    for req_id, spec_tokens in sched_spec.items():
        orig_budget = len(spec_tokens)
        num_invalid = num_invalid_map.get(req_id, 0)
        valid_len = max(0, orig_budget - num_invalid)
        raw_draft = list(spec_tokens[:valid_len])

        grammar = None
        remaining = None
        request = (
            scheduler.requests.get(req_id) if hasattr(scheduler, "requests") else None
        )
        if (
            request is not None
            and not request.is_finished()
            and not getattr(request, "is_prefill_chunk", False)
            and len(request.output_token_ids) > 0
        ):
            can_advance = not hasattr(
                manager, "should_advance"
            ) or manager.should_advance(request)
            structured_req = getattr(request, "structured_output_request", None)
            if can_advance and structured_req is not None:
                grammar = getattr(structured_req, "grammar", None)
                remaining = request.max_tokens - len(request.output_token_ids) - 1

        prior_flags = sched_flags.get(req_id)
        plan = CompositeDraftPlanner.plan(
            grammar=grammar,
            tokenizer=tokenizer,
            existing_model_draft=raw_draft,
            budget=orig_budget,
            max_remaining_tokens=remaining,
            stats=_STATS,
            existing_model_flags=prior_flags,
        )

        spec_tokens[:] = plan.tokens
        sched_flags[req_id] = plan.flags
        num_invalid_map[req_id] = plan.num_invalid


def topup_running_requests(scheduler: Any) -> None:
    """Fills each structured request's unused speculative slots using CompositeDraftPlanner."""  # noqa: E501
    if getattr(getattr(scheduler, "scheduler_config", None), "async_scheduling", False):
        return
    num_spec_tokens = getattr(scheduler, "num_spec_tokens", 0)
    if num_spec_tokens <= 0:
        return

    manager = getattr(scheduler, "structured_output_manager", None)
    if manager is None:
        return
    tokenizer = getattr(manager, "tokenizer", None)
    if tokenizer is None:
        return

    for request in scheduler.running:
        if (
            request.is_finished()
            or getattr(request, "is_prefill_chunk", False)
            or len(request.output_token_ids) == 0
        ):
            continue
        if not hasattr(request, "spec_token_ids") or request.spec_token_ids is None:
            request.spec_token_ids = []
        if hasattr(manager, "should_advance") and not manager.should_advance(request):
            continue
        structured_req = getattr(request, "structured_output_request", None)
        if structured_req is None:
            continue
        grammar = getattr(structured_req, "grammar", None)
        if grammar is None:
            continue

        raw_draft = list(request.spec_token_ids)
        remaining = request.max_tokens - len(request.output_token_ids) - 1
        plan = CompositeDraftPlanner.plan(
            grammar=grammar,
            tokenizer=tokenizer,
            existing_model_draft=raw_draft,
            budget=num_spec_tokens,
            max_remaining_tokens=remaining,
            stats=_STATS,
            existing_model_flags=getattr(request, "spec_token_flags", None),
        )

        valid_count = num_spec_tokens - plan.num_invalid
        request.spec_token_ids = plan.tokens[:valid_count]
        request.spec_token_flags = plan.flags[:valid_count]

    if _STATS.calls and (_STATS.calls % 10 == 0 or _STATS.hits > 0):
        logger.info("TPU grammar jump decoding: %s", _STATS.summary())


def patch_vllm_grammar_jump_decoding() -> None:
    """Wraps Scheduler speculative draft methods with GaaD jump decoding."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_ENABLE_JUMP_DECODING:
        logger.info(
            "TPU structured jump decoding disabled by VLLM_TPU_ENABLE_JUMP_DECODING=0."
        )
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_tpu_structured_jump_patch", False):
        return

    original_update_in_output = Scheduler.update_draft_token_ids_in_output

    def patched_update_in_output(self, scheduler_output):
        original_update_in_output(self, scheduler_output)
        topup_scheduled_spec_decode_tokens(self, scheduler_output)

    Scheduler.update_draft_token_ids_in_output = patched_update_in_output

    original_update = Scheduler.update_draft_token_ids

    def patched_update_draft_token_ids(self):
        original_update(self)
        topup_running_requests(self)

    Scheduler.update_draft_token_ids = patched_update_draft_token_ids

    original_schedule = Scheduler.schedule

    def patched_schedule(self):
        output = original_schedule(self)
        if output is not None and not getattr(
            self.scheduler_config, "async_scheduling", False
        ):
            flags_dict = {}
            for req in self.running:
                if hasattr(req, "spec_token_flags") and req.spec_token_flags:
                    flags_dict[req.request_id] = req.spec_token_flags
            output.scheduled_spec_decode_flags = flags_dict
        return output

    Scheduler.schedule = patched_schedule
    Scheduler._tpu_structured_jump_patch = True
    logger.info(
        "Applied TPU patch: structured jump decoding (GaaD) via CompositeDraftPlanner."
    )
