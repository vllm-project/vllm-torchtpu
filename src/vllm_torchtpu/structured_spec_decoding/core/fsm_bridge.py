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
"""FSM lookahead and token jump generation bridge for structured speculative decoding."""  # noqa: E501

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.structured_spec_decoding.constants import DRAFT_FLAG_GRAMMAR


@dataclass
class JumpStats:
    """Counters tracking grammar jump decoding activity."""

    calls: int = 0
    hits: int = 0
    tokens_proposed: int = 0
    truncations: int = 0
    fsm_rejections: int = 0

    def summary(self) -> str:
        hit_pct = 100.0 * self.hits / self.calls if self.calls else 0.0
        avg_tokens = self.tokens_proposed / self.hits if self.hits else 0.0
        return (
            f"calls={self.calls} hits={self.hits} ({hit_pct:.1f}%) "
            f"tokens={self.tokens_proposed} (avg={avg_tokens:.2f}) "
            f"truncations={self.truncations} fsm_rejections={self.fsm_rejections}"
        )


def propose_jump_topup(
    grammar: Any,
    tokenizer: Any,
    validated_draft: list[int],
    num_spec_tokens: int,
    stats: JumpStats | None = None,
) -> list[int]:
    """Returns deterministic grammar tokens to append after ``validated_draft``."""
    budget = num_spec_tokens - len(validated_draft)
    if budget <= 0:
        return []
    if grammar.is_terminated():
        return []

    matcher = grammar.matcher
    if stats is not None:
        stats.calls += 1

    advanced = 0
    try:
        for token in validated_draft:
            if not matcher.accept_token(token):
                return []
            advanced += 1

        jump_str = matcher.find_jump_forward_string()
        if not jump_str:
            return []

        candidate = tokenizer.encode(jump_str, add_special_tokens=False)
        if not candidate:
            return []
        if len(candidate) > budget:
            candidate = candidate[:budget]
            if stats is not None:
                stats.truncations += 1

        jumped = 0
        for token in candidate:
            if not matcher.accept_token(token):
                break
            jumped += 1
        advanced += jumped

        if stats is not None:
            if jumped < len(candidate):
                stats.fsm_rejections += 1
            if jumped:
                stats.hits += 1
                stats.tokens_proposed += jumped
        return candidate[:jumped]
    finally:
        if advanced:
            matcher.rollback(advanced)


def propose_jump_topup_with_flags(
    grammar: Any,
    tokenizer: Any,
    validated_draft: list[int],
    num_spec_tokens: int,
    stats: JumpStats | None = None,
) -> tuple[list[int], list[int]]:
    """Returns deterministic grammar tokens and their draft flags (flag == 2)."""
    tokens = propose_jump_topup(
        grammar=grammar,
        tokenizer=tokenizer,
        validated_draft=validated_draft,
        num_spec_tokens=num_spec_tokens,
        stats=stats,
    )
    flags = [DRAFT_FLAG_GRAMMAR] * len(tokens)
    return tokens, flags
