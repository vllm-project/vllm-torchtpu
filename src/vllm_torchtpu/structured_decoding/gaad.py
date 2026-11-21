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
"""Grammar-as-a-Draft (GaaD): grammar-forced tokens in speculative draft rows.

Where a structured-output request's grammar admits exactly one continuation
string, the tokens of that string are known before the model runs. GaaD plans
them into the request's draft row with flag ``GRAMMAR``: the rejection sampler
can accept a ``GRAMMAR`` slot without comparing it with the target, while a
rejected ``MODEL`` slot ahead of it still rejects it, as it rejects every later
slot.

This module holds the host-side planner: ``GrammarCursor`` walks the request's
grammar matcher along a row, ``plan_draft_row`` merges the drafter's proposal
with the forced tokens, ``plan_request`` replaces a request's drafts with its
planned row, and ``scheduled_flags`` trims each row's flags to the drafts the
scheduler scheduled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DRAFT_FLAG_EMPTY = 0
DRAFT_FLAG_MODEL = 1
DRAFT_FLAG_GRAMMAR = 2


class GrammarCursor:
    """Walks a grammar matcher along a draft row; ``rollback`` restores it.

    The forced token is the next token of the canonical encoding of the
    matcher's jump-forward string. The encoding is used whole or not at all: a
    jump whose encoding the grammar rejects anywhere forces nothing. It is
    encoded once per jump run and looked up again after a token off the run.
    """

    def __init__(self, matcher: Any, tokenizer: Any):
        self._matcher = matcher
        self._tokenizer = tokenizer
        self._advanced = 0
        # The rest of the current jump run, or None before it is looked up.
        self._run: list[int] | None = None

    def forced_next(self) -> int | None:
        """The only token the grammar lets the row continue with, or None."""
        if self._run is None:
            self._run = self._jump_tokens()
        return self._run[0] if self._run else None

    def advance(self, token: int) -> bool:
        """Advances past ``token`` if the grammar allows it."""
        if not self._matcher.accept_token(token):
            return False
        self._advanced += 1
        if self._run and self._run[0] == token:
            # An exhausted run is looked up again: the grammar may force more.
            self._run = self._run[1:] or None
        else:
            self._run = None
        return True

    def rollback(self) -> None:
        if self._advanced:
            self._matcher.rollback(self._advanced)
        self._advanced = 0
        self._run = None

    def _jump_tokens(self) -> list[int]:
        if self._matcher.is_terminated():
            return []
        jump = self._matcher.find_jump_forward_string()
        if not jump:
            return []
        tokens = list(self._tokenizer.encode(jump, add_special_tokens=False))
        probe = self._matcher.fork()
        if not all(probe.accept_token(token) for token in tokens):
            return []
        return tokens


def plan_draft_row(
    model_draft: Sequence[int], cursor: GrammarCursor, k: int
) -> tuple[list[int], list[int]]:
    """Merges ``model_draft`` with the tokens the grammar forces, ``k`` at most.

    1. A model token that is the forced token becomes a ``GRAMMAR`` slot; one
       the grammar allows where it forces nothing stays a ``MODEL`` slot.
    2. The model draft is cut at its first token the grammar rejects or that
       differs from the forced token: its later tokens continue from a
       tokenization the forced path does not take.
    3. Forced tokens fill the remaining slots while the grammar forces one.

    Advances ``cursor`` past every planned token. The row has no ``EMPTY``
    padding: the scheduler schedules exactly the planned tokens.
    """
    tokens: list[int] = []
    flags: list[int] = []
    for token in model_draft[:k]:
        forced = cursor.forced_next()
        if (forced is not None and forced != token) or not cursor.advance(token):
            break
        tokens.append(token)
        flags.append(DRAFT_FLAG_GRAMMAR if forced == token else DRAFT_FLAG_MODEL)
    while len(tokens) < k:
        forced = cursor.forced_next()
        if forced is None or not cursor.advance(forced):
            break
        tokens.append(forced)
        flags.append(DRAFT_FLAG_GRAMMAR)
    return tokens, flags


def _has_penalties(sampling_params: Any) -> bool:
    return (
        sampling_params.presence_penalty != 0.0
        or sampling_params.frequency_penalty != 0.0
        or sampling_params.repetition_penalty != 1.0
    )


def plan_request(
    request: Any, structured_output_manager: Any, k: int
) -> list[int] | None:
    """Replaces ``request.spec_token_ids`` with its GaaD row; returns its flags.

    Returns None, leaving the drafts as they are, for a request GaaD skips: one
    still prefilling, whose grammar does not constrain its next token (it has
    none, it is not xgrammar, or reasoning has not ended), or that reports
    logprobs or applies penalties (a ``GRAMMAR`` slot has no target logits row
    to report or penalize).
    """
    if request.is_prefill_chunk or request.num_output_placeholders:
        return None
    structured = request.structured_output_request
    if structured is None:
        return None
    # Reasoning tokens before the grammar starts are unconstrained.
    bounds = structured_output_manager._get_constraint_bounds(request, ())
    if bounds.constraint_start != 0:
        return None
    grammar = structured.grammar
    matcher = getattr(grammar, "matcher", None)
    if matcher is None or grammar.is_terminated():
        return None
    params = request.sampling_params
    if params is None or params.logprobs is not None or _has_penalties(params):
        return None
    budget = min(k, request.max_tokens - request.num_output_tokens - 1)
    if budget <= 0:
        return None
    cursor = GrammarCursor(matcher, structured_output_manager.tokenizer)
    try:
        tokens, flags = plan_draft_row(request.spec_token_ids, cursor, budget)
    finally:
        cursor.rollback()
    request.spec_token_ids = tokens
    return flags


def scheduled_flags(
    flags_by_req: dict[str, list[int]],
    scheduled_spec_decode_tokens: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Trims each planned row's flags to the drafts the scheduler scheduled."""
    return {
        req_id: flags_by_req[req_id][: len(tokens)]
        for req_id, tokens in scheduled_spec_decode_tokens.items()
        if req_id in flags_by_req
    }
