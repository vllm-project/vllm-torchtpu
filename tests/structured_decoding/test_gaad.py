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

import numpy as np
import pytest
import torch

from vllm_torchtpu.structured_decoding import gaad
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


def _vllm_config(speculative_config=None, async_scheduling=True, gaad=None):
    return SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        additional_config={} if gaad is None else {"gaad": gaad},
    )


_STANDALONE = SimpleNamespace(
    method="custom_class", model=gaad.GAAD_PROPOSER, num_speculative_tokens=7
)
_DRAFTER = SimpleNamespace(method="mtp", model=None, num_speculative_tokens=7)


def test_check_config_needs_gaad_for_standalone():
    with pytest.raises(ValueError, match="gaad entry"):
        gaad.check_config(_vllm_config(_STANDALONE))


def test_check_config_needs_a_speculative_config():
    with pytest.raises(ValueError, match="speculative config"):
        gaad.check_config(_vllm_config(gaad={}))


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (True, "must be a dict"),
        ({"draft_token": 3}, "Unknown"),
        ({"draft_tokens": 0}, "from 1 to"),
        ({"draft_tokens": 8}, "from 1 to"),
        ({"draft_tokens": "3"}, "from 1 to"),
        ({"draft_tokens": True}, "from 1 to"),
    ],
)
def test_check_config_rejects_bad_options(options, match):
    with pytest.raises(ValueError, match=match):
        gaad.check_config(_vllm_config(_DRAFTER, gaad=options))


def test_check_config_turns_async_scheduling_off():
    config = _vllm_config(_DRAFTER, async_scheduling=True, gaad={"draft_tokens": 3})
    gaad.check_config(config)
    assert config.scheduler_config.async_scheduling is False


def test_check_config_leaves_other_configs_alone():
    config = _vllm_config(SimpleNamespace(method="ngram", model=None))
    gaad.check_config(config)
    assert config.scheduler_config.async_scheduling is True


def _manager_for(req_ids, enabled):
    from vllm_torchtpu.runner.speculative_decoding_manager import (
        SpeculativeDecodingManager,
    )

    manager = SpeculativeDecodingManager.__new__(SpeculativeDecodingManager)
    manager.runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(req_ids=req_ids),
        vllm_config=_vllm_config(_DRAFTER, gaad={} if enabled else None),
    )
    return manager


@pytest.mark.parametrize("enabled", [False, True])
def test_draft_flags_are_laid_out_like_the_drafts(enabled):
    manager = _manager_for(["x", "a", "b", "c"], enabled)
    flat = manager._flat_draft_flags(
        np.array([3, 0, 2], dtype=np.int32), {"a": [G, G, M], "c": [G]}, start_index=1
    )
    # a: its flags; b: no drafts; c: G then MODEL by default.
    assert flat.tolist() == [G, G, M, G, M]
    flags = manager._build_draft_flags_tensor(flat, 8)
    if not enabled:
        assert flags is None
        return
    assert flags.tolist() == [G, G, M, G, M, 0, 0, 0]


def test_grammar_rows_skip_the_lm_head():
    from vllm_torchtpu.runner.speculative_decoding_manager import (
        SpeculativeDecodingManager,
    )

    # Request 0 drafts [G, M, G] (logits rows 0-2, bonus row 3); request 1
    # drafts [G, G] (rows 4-5, bonus row 6).
    logits, bonus, target = SpeculativeDecodingManager._skip_grammar_rows(
        np.array([3, 2]),
        np.array([G, M, G, G, G]),
        np.array([10, 11, 12, 13, 20, 21, 22]),
        np.array([3, 6]),
        np.array([0, 1, 2, 4, 5]),
    )
    # Only the MODEL draft's row and the bonus rows keep the LM head.
    assert logits.tolist() == [11, 13, 22]
    assert bonus.tolist() == [1, 2]
    # GRAMMAR drafts point at their request's bonus row.
    assert target.tolist() == [1, 0, 1, 2, 2]


def test_the_drafter_fills_only_its_share_of_the_row():
    assert gaad.draft_limit(_vllm_config(_DRAFTER, gaad={"draft_tokens": 1})) == 1
    # The drafter's x after the forced run is past its share; the grammar
    # still fills the row.
    request = _request(spec_token_ids=[1, 2, 3, 7])
    assert plan_request(request, _manager(), 7, 1) == [G, G, G]
    assert request.spec_token_ids == [1, 2, 3]


def test_a_skipped_request_keeps_the_drafters_share():
    skipped = _request(request_id="s", spec_token_ids=[7, 8, 9], is_prefill_chunk=True)
    scheduler = SimpleNamespace(
        structured_output_manager=_manager(),
        num_spec_tokens=7,
        running=[skipped],
        vllm_config=_vllm_config(_DRAFTER, gaad={"draft_tokens": 2}),
    )
    assert gaad.plan_running_requests(scheduler) == {}
    assert skipped.spec_token_ids == [7, 8]


def test_draft_limit_is_the_row_without_draft_tokens():
    assert gaad.draft_limit(_vllm_config(_DRAFTER)) == 7
    assert gaad.draft_limit(_vllm_config(_DRAFTER, gaad={})) == 7
