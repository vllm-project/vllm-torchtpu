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
"""CPU unit tests for the GaaD (Grammar-as-a-Draft) subsystem."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_torchtpu.spec_decode.gaad import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    DRAFT_FLAG_MODEL,
    CompositeDraftPlanner,
    FSMBridge,
    JumpStats,
    topup_running_requests,
    validate_draft_row_invariant,
)


class _FakeMatcher:
    def __init__(self, jump_str: str, allowed_tokens: set[int]):
        self.jump_str = jump_str
        self.allowed_tokens = allowed_tokens
        self.history: list[int] = []

    def find_jump_forward_string(self) -> str:
        return "" if self.history else self.jump_str

    def accept_token(self, token: int) -> bool:
        if token in self.allowed_tokens:
            self.history.append(token)
            return True
        return False

    def rollback(self, n: int) -> None:
        for _ in range(n):
            if self.history:
                self.history.pop()


class _FakeGrammar:
    def __init__(
        self, jump_str: str, allowed_tokens: set[int], terminated: bool = False
    ):
        self.matcher = _FakeMatcher(jump_str, allowed_tokens)
        self._terminated = terminated

    def is_terminated(self) -> bool:
        return self._terminated

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        out = []
        for t in tokens:
            if t in self.matcher.allowed_tokens:
                out.append(t)
            else:
                break
        return out


class _FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]):
        self.mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self.mapping.get(text, []))


def test_validate_draft_row_invariant():
    assert validate_draft_row_invariant([2, 2, 1, 2, 0, 0])
    assert validate_draft_row_invariant([0, 0, 0])
    assert validate_draft_row_invariant([2, 2, 2])
    assert not validate_draft_row_invariant([2, 0, 1])
    assert not validate_draft_row_invariant([0, 2])
    assert not validate_draft_row_invariant([3, 1])


def test_fsm_bridge_extracts_and_rolls_back():
    grammar = _FakeGrammar('{"name":"', {10, 11, 12})
    tokenizer = _FakeTokenizer({'{"name":"': [10, 11, 12]})
    stats = JumpStats()
    bridge = FSMBridge(stats=stats)

    tokens, flags = bridge.extract_deterministic_jump(
        grammar=grammar, tokenizer=tokenizer, num_spec_tokens=7
    )
    assert tokens == [10, 11, 12]
    assert flags == [DRAFT_FLAG_GRAMMAR] * 3
    assert grammar.matcher.history == []
    assert stats.hits == 1
    assert stats.tokens_proposed == 3


def test_composite_draft_planner_upgrades_leading_and_tops_up():
    grammar = _FakeGrammar('{"a":', {10, 11, 20})
    tokenizer = _FakeTokenizer({'{"a":': [10, 11]})
    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=[10, 20, 99],
        budget=5,
    )
    # 10 upgraded to GRAMMAR(2), 20 stays MODEL(1), 99 cut, rest EMPTY(0)
    assert plan.tokens == [10, 20, -1, -1, -1]
    assert plan.flags == [
        DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_MODEL,
        DRAFT_FLAG_EMPTY,
        DRAFT_FLAG_EMPTY,
        DRAFT_FLAG_EMPTY,
    ]
    assert plan.num_invalid == 3
    assert validate_draft_row_invariant(plan.flags)


def test_topup_running_requests_populates_spec_token_flags():
    grammar = _FakeGrammar('","b":"', {31, 32})
    tokenizer = _FakeTokenizer({'","b":"': [31, 32]})
    req = SimpleNamespace(
        request_id="r1",
        structured_output_request=SimpleNamespace(grammar=grammar),
        spec_token_ids=[],
        max_tokens=64,
        output_token_ids=[1, 2],
    )
    sched = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=4),
        structured_output_manager=SimpleNamespace(tokenizer=tokenizer),
        running=[req],
    )
    topup_running_requests(sched)
    assert req.spec_token_ids == [31, 32]
    assert req.spec_token_flags == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR]


def test_structured_decoding_manager_skips_grammar_slots():
    from vllm_torchtpu.structured_decoding.structured_decoding_manager import (
        StructuredDecodingManager,
    )

    runner = MagicMock()
    runner.vocab_size = 64
    runner.max_num_reqs = 4
    runner.device = torch.device("cpu")
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    runner._mtp_prefill_only = False
    runner.num_tokens_paddings = [4, 8, 16]
    runner.input_batch.req_id_to_index = {"r1": 0}

    mgr = StructuredDecodingManager(runner)
    bitmask = np.ones((4, 2), dtype=np.int32)
    grammar_out = SimpleNamespace(
        grammar_bitmask=bitmask,
        structured_output_request_ids=["r1"],
    )
    target_logits = torch.zeros((4, 64), dtype=torch.float32)
    bonus_logits = torch.zeros((1, 64), dtype=torch.float32)
    draft_lengths = np.array([3], dtype=np.int32)

    # All 3 drafts are GRAMMAR (2): target_bitmask should be dummy (require=False)
    req_t, t_mask, req_b, b_mask, _ = mgr.prepare_spec_structured_decoding_input(
        target_logits=target_logits,
        bonus_logits=bonus_logits,
        grammar_output=grammar_out,
        scheduled_spec_decode_tokens={"r1": [10, 11, 12]},
        draft_lengths_cpu=draft_lengths,
        cur_start_idx=0,
        cur_end_idx=1,
        scheduled_spec_decode_flags={"r1": [2, 2, 2]},
    )
    assert req_t is not None and not bool(req_t.any())
    assert bool(req_b[0].item())

    # Mixed [2, 1, 2]: only slot 1 (MODEL) requires target masking
    req_t2, _, _, _, _ = mgr.prepare_spec_structured_decoding_input(
        target_logits=target_logits,
        bonus_logits=bonus_logits,
        grammar_output=grammar_out,
        scheduled_spec_decode_tokens={"r1": [10, 11, 12]},
        draft_lengths_cpu=draft_lengths,
        cur_start_idx=0,
        cur_end_idx=1,
        scheduled_spec_decode_flags={"r1": [2, 1, 2]},
    )
    assert req_t2 is not None
    assert req_t2[:3].squeeze(-1).tolist() == [False, True, False]
