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
"""Tests for Standalone GaaD (Grammar-as-a-Draft) Speculative Decoding without external drafter models.

Verifies:
1. Standalone GaaD when FSM has a jump proposes purely GRAMMAR tokens (flag 2).
2. Standalone GaaD when FSM has NO jump proposes 0 draft tokens (so it falls back to 1-token decode).
3. GaaD pre-jump takes priority over model draft when pre-jump is available.
4. Topup works when request has no pre-existing spec_token_ids (pure GaaD in scheduler).
"""

from unittest.mock import MagicMock

from vllm_torchtpu.spec_decode.grammar_jump import (DRAFT_FLAG_EMPTY,
                                                    DRAFT_FLAG_GRAMMAR,
                                                    CompositeDraftPlanner,
                                                    _topup_running_requests)


class MockFSM:

    def __init__(self, transitions: dict[tuple[int, ...], str],
                 valid_tokens: set[int]):
        self.transitions = transitions
        self.valid_tokens = valid_tokens
        self.accepted: list[int] = []

    def accept_token(self, token_id: int) -> bool:
        if token_id not in self.valid_tokens:
            return False
        self.accepted.append(token_id)
        return True

    def find_jump_forward_string(self) -> str:
        state = tuple(self.accepted)
        return self.transitions.get(state, "")

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            del self.accepted[-num_tokens:]


class MockGrammar:

    def __init__(self, fsm: MockFSM):
        self.matcher = fsm

    def is_terminated(self) -> bool:
        return False

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        valid = []
        for t in tokens:
            if t in self.matcher.valid_tokens:
                valid.append(t)
            else:
                break
        return valid


class MockTokenizer:

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def test_standalone_gaad_with_jump():
    """When FSM has jump 'name', standalone GaaD proposes [110, 97, 109, 101] with flag GRAMMAR."""
    fsm = MockFSM(transitions={(): "name"}, valid_tokens={110, 97, 109, 101})
    grammar = MockGrammar(fsm)
    tokenizer = MockTokenizer()

    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=None,
        budget=7,
    )

    assert plan.tokens == [110, 97, 109, 101, -1, -1, -1]
    assert plan.flags == [
        DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_EMPTY, DRAFT_FLAG_EMPTY,
        DRAFT_FLAG_EMPTY
    ]
    assert plan.num_invalid == 3


def test_standalone_gaad_without_jump_yields_zero_valid_drafts():
    """When FSM has NO jump (free text), standalone GaaD proposes 0 draft tokens."""
    fsm = MockFSM(transitions={}, valid_tokens={10, 20, 30})
    grammar = MockGrammar(fsm)
    tokenizer = MockTokenizer()

    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=None,
        budget=7,
    )

    assert plan.tokens == [-1, -1, -1, -1, -1, -1, -1]
    assert plan.flags == [DRAFT_FLAG_EMPTY] * 7
    assert plan.num_invalid == 7


def test_gaad_standalone_mode_ignores_model_draft():
    """When standalone=True is set, GaaD ignores existing_model_draft completely."""
    fsm = MockFSM(transitions={(): "AB"}, valid_tokens={65, 66, 99})
    grammar = MockGrammar(fsm)
    tokenizer = MockTokenizer()

    # In standalone mode, even if existing_model_draft=[99] is passed, it is ignored
    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=[99],
        budget=5,
        standalone=True,
    )

    # GaaD pre-jumps [65, 66] as pure grammar tokens, model draft [99] ignored
    assert plan.tokens == [65, 66, -1, -1, -1]
    assert plan.flags == [
        DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_EMPTY,
        DRAFT_FLAG_EMPTY, DRAFT_FLAG_EMPTY
    ]


def test_standalone_topup_running_requests_initializes_spec_tokens():
    """Scheduler top-up must initialize request.spec_token_ids if not present (Pure GaaD)."""
    fsm = MockFSM(transitions={(): "hi"}, valid_tokens={104, 105})
    grammar = MockGrammar(fsm)
    tokenizer = MockTokenizer()

    request = MagicMock()
    request.is_finished.return_value = False
    request.is_prefill_chunk = False
    request.max_tokens = 50
    request.output_token_ids = [1]
    request.structured_output_request.grammar = grammar
    # simulate a request that was NOT touched by any model drafter
    if hasattr(request, "spec_token_ids"):
        delattr(request, "spec_token_ids")

    scheduler = MagicMock()
    scheduler.num_spec_tokens = 4
    scheduler.running = [request]
    scheduler.structured_output_manager.tokenizer = tokenizer
    scheduler.structured_output_manager.should_advance.return_value = True

    _topup_running_requests(scheduler)

    # Must be initialized and filled with [104, 105]
    assert hasattr(request, "spec_token_ids")
    assert request.spec_token_ids == [104, 105]
    assert request.spec_token_flags == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR]
