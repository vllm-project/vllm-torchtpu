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
"""CPU tests for GaaD (Grammar-as-a-Draft) draft-row planning."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from vllm_torchtpu.structured_decoding.gaad import (
    DRAFT_FLAG_GRAMMAR as G,
)
from vllm_torchtpu.structured_decoding.gaad import (
    DRAFT_FLAG_MODEL as M,
)
from vllm_torchtpu.structured_decoding.gaad import (
    GrammarCursor,
    plan_draft_row,
    plan_request,
    scheduled_flags,
)

# A JSON-like grammar `{"a":<free>}` whose one free slot takes "x" or "y".
VOCAB = {1: '{"', 2: "a", 3: '":', 4: '"', 5: ":", 7: "x", 8: "y", 9: "}", 10: "{"}
TEMPLATE = ['{"a":', None, "}"]
FREE = {7, 8}


class _Matcher:
    """Character-level matcher of TEMPLATE with xgrammar's matcher surface."""

    def __init__(self) -> None:
        self.part = 0
        self.offset = 0
        self.history: list[tuple[int, int]] = []

    def is_terminated(self) -> bool:
        return self.part == len(TEMPLATE)

    def find_jump_forward_string(self) -> str:
        if self.is_terminated() or TEMPLATE[self.part] is None:
            return ""
        return TEMPLATE[self.part][self.offset :]

    def accept_token(self, token: int) -> bool:
        state = (self.part, self.offset)
        if self.is_terminated():
            return False
        if TEMPLATE[self.part] is None:
            if token not in FREE:
                return False
            self.part, self.offset = self.part + 1, 0
        else:
            text = VOCAB.get(token, "")
            rest = TEMPLATE[self.part][self.offset :]
            if not text or not rest.startswith(text):
                return False
            self.offset += len(text)
            if self.offset == len(TEMPLATE[self.part]):
                self.part, self.offset = self.part + 1, 0
        self.history.append(state)
        return True

    def rollback(self, n: int) -> None:
        for _ in range(n):
            self.part, self.offset = self.history.pop()

    def fork(self) -> _Matcher:
        return copy.deepcopy(self)


class _Tokenizer:
    """Encodes by longest match over VOCAB, or by a fixed table."""

    def __init__(self, table: dict[str, list[int]] | None = None) -> None:
        self.table = table or {}
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        self.calls += 1
        if text in self.table:
            return list(self.table[text])
        tokens = []
        while text:
            tid, piece = max(
                ((t, s) for t, s in VOCAB.items() if text.startswith(s)),
                key=lambda ts: len(ts[1]),
            )
            tokens.append(tid)
            text = text[len(piece) :]
        return tokens


def _plan(model_draft, k=7, tokenizer=None, matcher=None):
    matcher = matcher or _Matcher()
    cursor = GrammarCursor(matcher, tokenizer or _Tokenizer())
    try:
        return plan_draft_row(model_draft, cursor, k)
    finally:
        cursor.rollback()


def test_grammar_fills_a_row_without_a_drafter():
    assert _plan([]) == ([1, 2, 3], [G, G, G])


def test_row_stops_at_the_budget():
    assert _plan([], k=2) == ([1, 2], [G, G])


def test_forced_model_tokens_are_upgraded_and_free_ones_stay_model():
    # x at the free slot stays MODEL; the "}" after it is forced again.
    assert _plan([1, 2, 3, 7, 9]) == ([1, 2, 3, 7, 9], [G, G, G, M, G])


def test_grammar_fills_after_a_model_token():
    assert _plan([1, 2, 3, 8]) == ([1, 2, 3, 8, 9], [G, G, G, M, G])


def test_a_non_canonical_model_tokenization_is_cut():
    # '{' '"' 'a' spells the forced text but not its tokens: the row takes the
    # forced tokens instead of continuing from the drafter's tokenization.
    assert _plan([10, 4, 2, 3, 7]) == ([1, 2, 3], [G, G, G])


def test_a_model_token_the_grammar_rejects_is_cut():
    assert _plan([1, 2, 3, 9]) == ([1, 2, 3], [G, G, G])


def test_a_jump_the_grammar_rejects_anywhere_forces_nothing():
    # 'y' cannot follow '{"': the whole encoding is dropped, not its prefix.
    tokenizer = _Tokenizer({'{"a":': [1, 8, 3]})
    assert _plan([], tokenizer=tokenizer) == ([], [])


def test_planning_restores_the_matcher():
    matcher = _Matcher()
    _plan([1, 2, 3, 7], matcher=matcher)
    assert (matcher.part, matcher.offset, matcher.history) == (0, 0, [])


def test_a_jump_run_is_encoded_once():
    tokenizer = _Tokenizer()
    _plan([], tokenizer=tokenizer)
    # One encode for the run; looking past its end finds the free slot.
    assert tokenizer.calls == 1


def _request(**overrides):
    params = SimpleNamespace(
        logprobs=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    grammar = SimpleNamespace(matcher=_Matcher(), is_terminated=lambda: False)
    fields = dict(
        request_id="r",
        is_prefill_chunk=False,
        num_output_placeholders=0,
        structured_output_request=SimpleNamespace(grammar=grammar),
        sampling_params=params,
        max_tokens=100,
        num_output_tokens=0,
        spec_token_ids=[],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _manager(constraint_start=0):
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        _get_constraint_bounds=lambda request, spec: SimpleNamespace(
            constraint_start=constraint_start
        ),
    )


def test_plan_request_replaces_the_drafts():
    request = _request(spec_token_ids=[1, 2])
    assert plan_request(request, _manager(), 7) == [G, G, G]
    assert request.spec_token_ids == [1, 2, 3]


def test_plan_request_leaves_the_last_output_token_out_of_the_budget():
    request = _request(max_tokens=3, num_output_tokens=1)
    assert plan_request(request, _manager(), 7) == [G]
    assert request.spec_token_ids == [1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_prefill_chunk": True},
        {"num_output_placeholders": 1},
        {"structured_output_request": None},
        {"sampling_params": SimpleNamespace(logprobs=1)},
        {
            "sampling_params": SimpleNamespace(
                logprobs=None,
                presence_penalty=0.5,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
            )
        },
        {"max_tokens": 1},
    ],
)
def test_plan_request_skips(overrides):
    request = _request(spec_token_ids=[7], **overrides)
    assert plan_request(request, _manager(), 7) is None
    assert request.spec_token_ids == [7]


def test_plan_request_skips_while_reasoning():
    request = _request(spec_token_ids=[7])
    assert plan_request(request, _manager(constraint_start=1), 7) is None


def test_plan_request_skips_a_terminated_grammar():
    grammar = SimpleNamespace(matcher=_Matcher(), is_terminated=lambda: True)
    request = _request(structured_output_request=SimpleNamespace(grammar=grammar))
    assert plan_request(request, _manager(), 7) is None


def test_scheduled_flags_follow_the_scheduled_drafts():
    flags = scheduled_flags({"a": [G, G, M], "b": [G]}, {"a": [1, 2], "c": [5, 6]})
    assert flags == {"a": [G, G]}
