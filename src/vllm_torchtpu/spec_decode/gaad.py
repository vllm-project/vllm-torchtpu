# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Grammar-as-a-Draft (GaaD) speculative decoding subsystem.

Encapsulates:
1. Flag constants (EMPTY=0, MODEL=1, GRAMMAR=2) and validate_draft_row_invariant().
2. FSMBridge for sub-5us deterministic lookahead and BPE boundary validation.
3. CompositeDraftPlanner (Pre-Jump, Model Absorption & Upgrade, Post-TopUp, Pad).
4. Scheduler lifecycle hooks attaching scheduled_spec_decode_flags.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("vllm." + __name__)

DRAFT_FLAG_EMPTY = 0
DRAFT_FLAG_MODEL = 1
DRAFT_FLAG_GRAMMAR = 2


@dataclass
class JumpStats:
    """Opportunity and yield counters for GaaD top-up."""

    calls: int = 0
    hits: int = 0
    tokens_proposed: int = 0
    truncations: int = 0
    fsm_rejections: int = 0

    def summary(self) -> str:
        hit_rate = (self.hits / self.calls) if self.calls else 0.0
        per_hit = (self.tokens_proposed / self.hits) if self.hits else 0.0
        return (
            f"calls={self.calls} hits={self.hits} "
            f"hit_rate={hit_rate:.3f} tokens={self.tokens_proposed} "
            f"tokens_per_hit={per_hit:.2f} "
            f"truncations={self.truncations} "
            f"fsm_rejections={self.fsm_rejections}"
        )


_STATS = JumpStats()


def get_stats() -> JumpStats:
    """Returns the process-local counters."""
    return _STATS


def validate_draft_row_invariant(flags: Sequence[int]) -> bool:
    """Validates one row of draft flags against the hardware contract.

    Grammar (2) and model (1) slots may interleave freely; only EMPTY (0)
    slots must form a contiguous suffix.
    """
    valid_flags = {DRAFT_FLAG_EMPTY, DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR}
    seen_empty = False
    for flag in flags:
        if flag not in valid_flags:
            return False
        if flag == DRAFT_FLAG_EMPTY:
            seen_empty = True
        elif seen_empty:
            return False
    return True


class FSMBridge:
    """Extracts deterministic lookahead tokens from the grammar FSM."""

    def __init__(self, stats: JumpStats | None = None):
        self.stats = stats

    def extract_deterministic_jump(
        self,
        grammar: Any,
        tokenizer: Any,
        validated_draft: list[int] | None = None,
        num_spec_tokens: int = 0,
    ) -> tuple[list[int], list[int]]:
        """Extracts deterministic jump tokens and flags, restoring FSM state."""
        if validated_draft is None:
            validated_draft = []
        budget = num_spec_tokens - len(validated_draft)
        if budget <= 0 or grammar is None or grammar.is_terminated():
            return [], []

        matcher = getattr(grammar, "matcher", None)
        if matcher is None:
            return [], []

        if self.stats is not None:
            self.stats.calls += 1

        advanced = 0
        try:
            for token in validated_draft:
                if not matcher.accept_token(token):
                    return [], []
                advanced += 1

            jump_str = matcher.find_jump_forward_string()
            if not jump_str:
                return [], []

            candidate = tokenizer.encode(jump_str, add_special_tokens=False)
            if not candidate:
                return [], []
            if len(candidate) > budget:
                candidate = candidate[:budget]
                if self.stats is not None:
                    self.stats.truncations += 1

            jumped = 0
            for token in candidate:
                if not matcher.accept_token(token):
                    break
                jumped += 1
            advanced += jumped

            if self.stats is not None:
                if jumped < len(candidate):
                    self.stats.fsm_rejections += 1
                if jumped:
                    self.stats.hits += 1
                    self.stats.tokens_proposed += jumped

            return candidate[:jumped], [DRAFT_FLAG_GRAMMAR] * jumped
        finally:
            if advanced:
                matcher.rollback(advanced)


@dataclass
class CompositeDraftPlan:
    """Integrated speculative draft plan."""

    tokens: list[int]
    flags: list[int]
    num_invalid: int


class CompositeDraftPlanner:
    """Unifies Pre-Jump, Model Absorption & Upgrade, Post-TopUp, and Padding."""

    @classmethod
    def _resolve_model_flags(
        cls,
        grammar: Any,
        tokenizer: Any,
        grammar_usable: bool,
        valid_draft: list[int],
        existing_model_flags: list[int] | None,
        effective_budget: int,
    ) -> list[int]:
        if existing_model_flags:
            draft_flags = [
                f if f in (DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR) else DRAFT_FLAG_MODEL
                for f in existing_model_flags[: len(valid_draft)]
            ]
            if len(draft_flags) < len(valid_draft):
                draft_flags.extend(
                    [DRAFT_FLAG_MODEL] * (len(valid_draft) - len(draft_flags))
                )
        else:
            draft_flags = [DRAFT_FLAG_MODEL] * len(valid_draft)

        if grammar_usable and valid_draft:
            lead_bridge = FSMBridge(stats=None)
            lead_jump, _ = lead_bridge.extract_deterministic_jump(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=min(len(valid_draft), effective_budget),
            )
            for idx, tok in enumerate(valid_draft):
                if idx < len(lead_jump) and tok == lead_jump[idx]:
                    draft_flags[idx] = DRAFT_FLAG_GRAMMAR
                else:
                    break
        return draft_flags

    @classmethod
    def plan(
        cls,
        grammar: Any,
        tokenizer: Any,
        existing_model_draft: list[int] | None = None,
        budget: int = 7,
        max_remaining_tokens: int | None = None,
        stats: JumpStats | None = None,
        existing_model_flags: list[int] | None = None,
        standalone: bool = False,
    ) -> CompositeDraftPlan:
        effective_budget = budget
        if max_remaining_tokens is not None and max_remaining_tokens >= 0:
            effective_budget = min(budget, max_remaining_tokens)

        tokens: list[int] = []
        flags: list[int] = []
        grammar_usable = (
            grammar is not None
            and tokenizer is not None
            and not grammar.is_terminated()
        )
        bridge = FSMBridge(stats=stats)

        if standalone:
            existing_model_draft = None

        # Phase 1: Pre-Jump
        if not existing_model_draft and grammar_usable:
            pre_jump, pre_flags = bridge.extract_deterministic_jump(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=effective_budget,
            )
            if pre_jump:
                tokens.extend(pre_jump)
                flags.extend(pre_flags)

        # Phase 2: Model Draft absorption & leading-run upgrade
        if existing_model_draft:
            remaining_for_model = effective_budget - len(tokens)
            if remaining_for_model > 0:
                raw_draft = list(existing_model_draft[:remaining_for_model])
                valid_draft = (
                    grammar.validate_tokens(raw_draft)
                    if grammar is not None
                    else raw_draft
                )
                draft_flags = cls._resolve_model_flags(
                    grammar=grammar,
                    tokenizer=tokenizer,
                    grammar_usable=grammar_usable,
                    valid_draft=valid_draft,
                    existing_model_flags=existing_model_flags,
                    effective_budget=effective_budget,
                )
                tokens.extend(valid_draft)
                flags.extend(draft_flags)

        # Phase 3: Post-TopUp
        remaining_for_topup = effective_budget - len(tokens)
        if remaining_for_topup > 0 and grammar_usable:
            topup, topup_flags = bridge.extract_deterministic_jump(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=tokens,
                num_spec_tokens=effective_budget,
            )
            if topup:
                tokens.extend(topup)
                flags.extend(topup_flags)

        # Phase 4: Static padding to budget
        num_valid = len(tokens)
        pad_len = max(0, budget - num_valid)
        tokens.extend([-1] * pad_len)
        flags.extend([DRAFT_FLAG_EMPTY] * pad_len)

        return CompositeDraftPlan(
            tokens=tokens[:budget],
            flags=flags[:budget],
            num_invalid=pad_len,
        )


def _get_scheduler_tokenizer(scheduler: Any) -> Any:
    structured_mgr = getattr(scheduler, "structured_output_manager", None)
    if structured_mgr is not None:
        tok = getattr(structured_mgr, "tokenizer", None)
        if tok is not None:
            return tok
    return getattr(scheduler, "tokenizer", None)


def _get_num_spec_tokens(scheduler: Any) -> int:
    spec_cfg = getattr(scheduler, "speculative_config", None)
    if spec_cfg is None:
        vllm_cfg = getattr(scheduler, "vllm_config", None)
        spec_cfg = getattr(vllm_cfg, "speculative_config", None) if vllm_cfg else None
    if spec_cfg is None:
        return 0
    return int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)


def topup_running_requests(scheduler: Any) -> None:
    """Applies CompositeDraftPlanner to running structured requests."""
    num_spec_tokens = _get_num_spec_tokens(scheduler)
    if num_spec_tokens <= 0:
        return
    tokenizer = _get_scheduler_tokenizer(scheduler)
    if tokenizer is None:
        return

    for request in getattr(scheduler, "running", ()):
        structured_req = getattr(request, "structured_output_request", None)
        if structured_req is None:
            continue
        grammar = getattr(structured_req, "grammar", None)
        if grammar is None:
            continue

        raw_draft = list(getattr(request, "spec_token_ids", ()) or ())
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


def topup_scheduled_spec_decode_tokens(scheduler: Any, scheduler_output: Any) -> None:
    """Applies CompositeDraftPlanner to scheduled_spec_decode_tokens in-place."""
    if scheduler_output is None:
        return
    sched_tokens = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
    if not sched_tokens:
        return
    num_spec_tokens = _get_num_spec_tokens(scheduler)
    if num_spec_tokens <= 0:
        return
    tokenizer = _get_scheduler_tokenizer(scheduler)
    if tokenizer is None:
        return

    requests_map = getattr(scheduler, "requests", {})
    sched_flags = getattr(scheduler_output, "scheduled_spec_decode_flags", None)
    if sched_flags is None:
        sched_flags = {}

    for req_id, spec_tokens in list(sched_tokens.items()):
        request = requests_map.get(req_id)
        if request is None:
            continue
        structured_req = getattr(request, "structured_output_request", None)
        if structured_req is None:
            continue
        grammar = getattr(structured_req, "grammar", None)
        if grammar is None:
            continue

        remaining = request.max_tokens - len(request.output_token_ids) - 1
        plan = CompositeDraftPlanner.plan(
            grammar=grammar,
            tokenizer=tokenizer,
            existing_model_draft=list(spec_tokens),
            budget=num_spec_tokens,
            max_remaining_tokens=remaining,
            stats=_STATS,
            existing_model_flags=sched_flags.get(req_id),
        )
        valid_count = num_spec_tokens - plan.num_invalid
        if valid_count > 0:
            sched_tokens[req_id] = plan.tokens[:valid_count]
            sched_flags[req_id] = plan.flags[:valid_count]

    if sched_flags:
        scheduler_output.scheduled_spec_decode_flags = sched_flags


def patch_vllm_grammar_jump_decoding() -> None:
    """Wraps Scheduler draft update and scheduling with GaaD top-up and flags."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_JUMP_DECODING:
        return

    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler

    if not hasattr(SchedulerOutput, "scheduled_spec_decode_flags"):
        SchedulerOutput.scheduled_spec_decode_flags = None

    if getattr(Scheduler, "_tpu_grammar_jump_patched", False):
        return

    original_update = Scheduler.update_draft_token_ids
    original_update_in_output = getattr(
        Scheduler, "update_draft_token_ids_in_output", None
    )
    original_schedule = getattr(Scheduler, "schedule", None)

    def update_draft_token_ids(
        self: Any, draft_token_ids: Any, *args: Any, **kwargs: Any
    ) -> None:
        original_update(self, draft_token_ids, *args, **kwargs)
        topup_running_requests(self)

    def update_draft_token_ids_in_output(
        self: Any,
        draft_token_ids: Any,
        scheduler_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if original_update_in_output is not None:
            original_update_in_output(
                self, draft_token_ids, scheduler_output, *args, **kwargs
            )
        topup_scheduled_spec_decode_tokens(self, scheduler_output)

    Scheduler.update_draft_token_ids = update_draft_token_ids
    if original_update_in_output is not None:
        Scheduler.update_draft_token_ids_in_output = update_draft_token_ids_in_output

    if original_schedule is not None:

        def schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
            req_flags: dict[str, list[int]] = {}
            for r in getattr(self, "running", ()):
                flags = getattr(r, "spec_token_flags", None)
                if flags:
                    req_flags[r.request_id] = list(flags)
                    r.spec_token_flags = None
            out = original_schedule(self, *args, **kwargs)
            if req_flags and hasattr(out, "scheduled_spec_decode_tokens"):
                sched_flags: dict[str, list[int]] = {}
                for req_id, spec_tokens in out.scheduled_spec_decode_tokens.items():
                    if req_id in req_flags:
                        f = req_flags[req_id][: len(spec_tokens)]
                        if len(f) < len(spec_tokens):
                            f.extend([DRAFT_FLAG_MODEL] * (len(spec_tokens) - len(f)))
                        sched_flags[req_id] = f
                if sched_flags:
                    out.scheduled_spec_decode_flags = sched_flags
            return out

        Scheduler.schedule = schedule

    Scheduler._tpu_grammar_jump_patched = True
    logger.info("Applied TPU patch: grammar jump decoding draft top-up.")


def patch_vllm_speculative_config_gaad() -> None:
    """Allows SpeculativeConfig to accept method='gaad'."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_JUMP_DECODING:
        return

    try:
        from vllm.config.speculative import SpeculativeConfig
    except (ImportError, ModuleNotFoundError):
        return

    if getattr(SpeculativeConfig, "_tpu_gaad_patched", False):
        return

    original_init = SpeculativeConfig.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        method = kwargs.get("method")
        if method == "gaad":
            kwargs["method"] = "ngram"
            original_init(self, *args, **kwargs)
            self.method = "gaad"
        else:
            original_init(self, *args, **kwargs)

    SpeculativeConfig.__init__ = patched_init
    SpeculativeConfig._tpu_gaad_patched = True
    logger.info("Applied TPU patch: SpeculativeConfig supports method='gaad'.")
