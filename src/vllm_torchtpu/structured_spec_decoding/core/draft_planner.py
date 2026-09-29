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
"""Universal speculative draft planning for structured speculative decoding (GaaD)."""

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.structured_spec_decoding.constants import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    DRAFT_FLAG_MODEL,
)
from vllm_torchtpu.structured_spec_decoding.core.fsm_bridge import (
    JumpStats,
    propose_jump_topup,
    propose_jump_topup_with_flags,
)


@dataclass(frozen=True)
class CompositeDraftPlan:
    """The outcome of one CompositeDraftPlanner pass."""

    tokens: list[int]
    """[budget] Draft token IDs, padded with -1."""

    flags: list[int]
    """[budget] Per-slot draft flags (0=EMPTY, 1=MODEL, 2=GRAMMAR)."""

    num_invalid: int
    """Count of empty/padding slots (-1) at the tail of the budget."""


class CompositeDraftPlanner:
    """Universal Speculative Drafter Planner.

    Unifies every grammar/speculative combination into a single pass that emits
    one row of ``budget`` slots, each slot carrying a *provenance* flag:
    - ``DRAFT_FLAG_GRAMMAR`` (2): forced-accepted during verification.
    - ``DRAFT_FLAG_MODEL`` (1): ordinary speculative guess, verified normally.
    - ``DRAFT_FLAG_EMPTY`` (0): padding (-1) up to static budget K.
    """

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

        if standalone:
            existing_model_draft = None

        # Phase 1: Pre-Jump (used for standalone GaaD or cold start)
        if not existing_model_draft and grammar_usable:
            pre_jump, pre_flags = propose_jump_topup_with_flags(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=effective_budget,
                stats=stats,
            )
            if pre_jump:
                tokens.extend(pre_jump)
                flags.extend(pre_flags)

        # Phase 2: Model Draft absorption & FSM validation
        if existing_model_draft:
            remaining_for_model = effective_budget - len(tokens)
            if remaining_for_model > 0:
                raw_draft = list(existing_model_draft[:remaining_for_model])
                if grammar is not None:
                    valid_draft = grammar.validate_tokens(raw_draft)
                else:
                    valid_draft = raw_draft
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

        # Phase 3: Post-TopUp (fill vacant speculative slots with grammar tokens)
        remaining_for_topup = effective_budget - len(tokens)
        if remaining_for_topup > 0 and grammar_usable:
            topup, topup_flags = propose_jump_topup_with_flags(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=tokens,
                num_spec_tokens=effective_budget,
                stats=stats,
            )
            if topup:
                tokens.extend(topup)
                flags.extend(topup_flags)

        # Phase 4: Static padding to budget K for TPU SIMD alignment
        num_valid = len(tokens)
        pad_len = max(0, budget - num_valid)
        if pad_len > 0:
            tokens.extend([-1] * pad_len)
            flags.extend([DRAFT_FLAG_EMPTY] * pad_len)

        return CompositeDraftPlan(
            tokens=tokens,
            flags=flags,
            num_invalid=pad_len,
        )

    @staticmethod
    def _resolve_model_flags(
        grammar: Any,
        tokenizer: Any,
        grammar_usable: bool,
        valid_draft: list[int],
        existing_model_flags: list[int] | None,
        effective_budget: int,
    ) -> list[int]:
        flags = [DRAFT_FLAG_MODEL] * len(valid_draft)
        if not valid_draft:
            return flags

        if existing_model_flags:
            for i, flag in enumerate(existing_model_flags[: len(valid_draft)]):
                if flag == DRAFT_FLAG_GRAMMAR:
                    flags[i] = DRAFT_FLAG_GRAMMAR

        if grammar_usable:
            forced = propose_jump_topup(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=effective_budget,
                stats=None,
            )
            for i in range(min(len(forced), len(valid_draft))):
                if forced[i] != valid_draft[i]:
                    break
                flags[i] = DRAFT_FLAG_GRAMMAR

        return flags
