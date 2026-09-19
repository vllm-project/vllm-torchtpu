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
"""Comprehensive unit tests for the unified CompositeDraftPlanner architecture.

Verifies:
1. Case 1 & 2: Post-Model Top-Up (D^b G^a 0^c) - N-gram / Eagle3 draft validated, then topped up with GaaD.
2. Case 3: Pre-Jump (G^a D^b 0^c) - GaaD pre-jumps before model draft.
3. Case 4: Full Composite (G^a D^b G^c 0^d) - Pre-jump, valid model draft, and subsequent top-up.
4. Model invalid draft trimming - FSM syntax violation truncates model draft and tops up.
5. Static TPU SIMD padding - Padding empty slots with -1 and DRAFT_FLAG_EMPTY (0).
"""

from vllm_torchtpu.spec_decode.grammar_jump import (DRAFT_FLAG_EMPTY,
                                                    DRAFT_FLAG_GRAMMAR,
                                                    DRAFT_FLAG_MODEL,
                                                    CompositeDraftPlanner)


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


def test_post_model_topup_case():
    """Case 1 & 2: Model proposed [10, 11] (e.g. Ngram or Eagle3), GaaD tops up with 'ok' [111, 107]."""
    # At state (10, 11), FSM can jump "ok"
    fsm = MockFSM(transitions={(10, 11): "ok"},
                  valid_tokens={10, 11, 111, 107})
    grammar = MockGrammar(fsm)
    tokenizer = MockTokenizer()

    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=[10, 11],
        budget=6,
    )

    # Output: [10, 11] (model) + [111, 107] (grammar) + [-1, -1] (padding)
    assert plan.tokens == [10, 11, 111, 107, -1, -1]
    assert plan.flags == [
        DRAFT_FLAG_MODEL, DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_EMPTY, DRAFT_FLAG_EMPTY
    ]
    assert plan.num_invalid == 2


def test_pre_jump_case():
    """Case 3: Pre-Jump: FSM jumps 'go' [103, 111] before model proposes."""
    # At base state (), FSM can jump "go"
    fsm = MockFSM(transitions={(): "go"}, valid_tokens={103, 111})
    grammar = MockGrammar(fsm)
    tokenizer = MockTokenizer()

    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=None,  # No model draft yet
        budget=5,
    )

    # Output: [103, 111] (grammar jump) + [-1, -1, -1] (remaining budget for model or padding)
    assert plan.tokens == [103, 111, -1, -1, -1]
    assert plan.flags == [
        DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_EMPTY,
        DRAFT_FLAG_EMPTY, DRAFT_FLAG_EMPTY
    ]
    assert plan.num_invalid == 3


def test_model_draft_with_syntax_violation_trimmed_and_topped_up():
    """Model proposed [10, 999, 998], where 999 is invalid.

    Trim at 10, then top up with 'ok' [111, 107].
    """
    fsm = MockFSM(transitions={(10, ): "ok"}, valid_tokens={10, 111, 107})
    grammar = MockGrammar(fsm)
    tokenizer = MockTokenizer()

    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=[10, 999, 998],
        budget=5,
    )

    # Output: [10] (model) + [111, 107] (grammar) + [-1, -1] (padding)
    assert plan.tokens == [10, 111, 107, -1, -1]
    assert plan.flags == [
        DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_EMPTY, DRAFT_FLAG_EMPTY
    ]
    assert plan.num_invalid == 2


def test_full_composite_pre_jump_and_post_topup():
    """Full composite multi-phase: FSM pre-jumps 'A' [65], model draft proposed [10], then FSM tops up 'Z' [90]."""
    fsm = MockFSM(
        transitions={
            (): "A",
            (65, 10): "Z"
        },
        valid_tokens={65, 10, 90},
    )
    grammar = MockGrammar(fsm)
    tokenizer = MockTokenizer()

    # Pre-jump generates [65]
    plan_pre = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=None,
        budget=5,
    )
    assert plan_pre.tokens[:1] == [65]

    # Next, model proposed [10] following pre-jump [65], and FSM tops up [90].
    # Slot 0 is the pre-jump token coming back around: the FSM still forces "A"
    # at this position, so it is recognised as GRAMMAR rather than demoted to
    # MODEL. Slot 1 is the genuine model guess, slot 2 the fresh top-up.
    plan_full = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=[65, 10],
        budget=5,
    )
    assert plan_full.tokens == [65, 10, 90, -1, -1]
    assert plan_full.flags == [
        DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_EMPTY, DRAFT_FLAG_EMPTY
    ]
    assert plan_full.num_invalid == 2
