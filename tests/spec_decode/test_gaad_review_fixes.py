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
"""Regression tests for defects found while reviewing the GaaD架构 (PR #1054).

These are pure-Python / CPU tensor tests: none of them touch the TPU sampler
kernels, so they run on a plain workstation. Each test below was RED before the
corresponding fix landed.

Covered defects:
1. ``_build_draft_flags_tensor`` ignored ``start_index`` and therefore read the
   flags of the wrong requests whenever ``_prepare_inputs`` chunked the batch.
2. ``CompositeDraftPlanner.plan`` could not express interleaved grammar/model
   rows: pre-existing GRAMMAR provenance was downgraded to MODEL on the
   round-trip, and a model draft that happens to reproduce a grammar-forced
   prefix was never recognised as grammar.
3. The row invariant rejected ``[1, 2, 2, 0, 0]``, a row ``plan()`` actually
   emits.
4. ``_topup_draft_token_ids`` was dead code with a padding convention that
   contradicted the live path.
"""

import types

import numpy as np
import torch

from vllm_torchtpu.spec_decode import grammar_jump
from vllm_torchtpu.spec_decode.grammar_jump import (
    DRAFT_FLAG_EMPTY, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_MODEL,
    CompositeDraftPlanner, validate_draft_row_invariant)

# ---------------------------------------------------------------------------
# Mocks (kept local on purpose: cross-importing test helpers couples unrelated
# test files together).
# ---------------------------------------------------------------------------


class MockFSM:
    """Minimal xgrammar matcher stand-in keyed by the accepted-token tuple."""

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
        return self.transitions.get(tuple(self.accepted), "")

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


# ---------------------------------------------------------------------------
# Defect 1: start_index
# ---------------------------------------------------------------------------


def test_build_draft_flags_tensor_respects_start_index():
    """Chunked ``_prepare_inputs`` calls must read flags from the right rows.

    ``tpu_runner`` slices ``req_ids[start_index:start_index + num_reqs]`` when
    building ``num_draft_tokens``. If the flag builder restarts from index 0 it
    pairs request r1's draft length with request r0's flags, which can plant a
    GRAMMAR flag on a model slot and force-accept a token the target rejected.
    """
    from vllm_torchtpu.runner.speculative_decoding_manager import \
        SpeculativeDecodingManager

    fake_self = types.SimpleNamespace(runner=types.SimpleNamespace(
        input_batch=types.SimpleNamespace(req_ids=["r0", "r1", "r2"]),
        device=torch.device("cpu"),
    ))
    # This chunk only covers r1 and r2.
    num_draft_tokens = np.array([2, 2], dtype=np.int32)
    flags = {
        "r0": [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR],
        "r1": [DRAFT_FLAG_MODEL, DRAFT_FLAG_MODEL],
        "r2": [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_MODEL],
    }

    out = SpeculativeDecodingManager._build_draft_flags_tensor(
        fake_self,
        num_draft_tokens=num_draft_tokens,
        padded_logits_length=8,
        scheduled_spec_decode_flags=flags,
        start_index=1,
    )

    assert out is not None
    assert out.tolist()[:4] == [
        DRAFT_FLAG_MODEL, DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_MODEL
    ]


def test_build_draft_flags_tensor_start_index_defaults_to_zero():
    """The unchunked path must keep working unchanged."""
    from vllm_torchtpu.runner.speculative_decoding_manager import \
        SpeculativeDecodingManager

    fake_self = types.SimpleNamespace(runner=types.SimpleNamespace(
        input_batch=types.SimpleNamespace(req_ids=["r0", "r1"]),
        device=torch.device("cpu"),
    ))
    out = SpeculativeDecodingManager._build_draft_flags_tensor(
        fake_self,
        num_draft_tokens=np.array([1, 1], dtype=np.int32),
        padded_logits_length=4,
        scheduled_spec_decode_flags={
            "r0": [DRAFT_FLAG_GRAMMAR],
            "r1": [DRAFT_FLAG_MODEL],
        },
    )
    assert out is not None
    assert out.tolist()[:2] == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_MODEL]


# ---------------------------------------------------------------------------
# Defect 3: the row invariant
# ---------------------------------------------------------------------------


def test_row_invariant_accepts_grammar_after_model():
    """Grammar tokens legitimately follow model tokens.

    ``plan()`` emits ``[MODEL, GRAMMAR, GRAMMAR, EMPTY, EMPTY]`` for the
    post-model top-up case, and the rejection kernel handles arbitrary
    interleaving because ``is_grammar | accepted`` is OR'd *before* the
    first-mismatch min-reduce. The only real constraint is that the empty slots
    form a suffix.
    """
    assert validate_draft_row_invariant([1, 2, 2, 0, 0]) is True
    assert validate_draft_row_invariant([2, 2, 1, 1, 2, 0]) is True
    assert validate_draft_row_invariant([2, 2, 0, 0]) is True
    assert validate_draft_row_invariant([1, 1, 1]) is True
    assert validate_draft_row_invariant([]) is True


def test_row_invariant_rejects_holes_and_bad_flags():
    """Empty slots must be a suffix, and flags must be in {0, 1, 2}."""
    assert validate_draft_row_invariant([2, 0, 1]) is False
    assert validate_draft_row_invariant([0, 1, 0]) is False
    assert validate_draft_row_invariant([1, 0, 2, 0]) is False
    assert validate_draft_row_invariant([3, 0]) is False
    assert validate_draft_row_invariant([-1, 1]) is False


# ---------------------------------------------------------------------------
# Defect 2: interleaved provenance
# ---------------------------------------------------------------------------


def test_model_draft_matching_grammar_prefix_is_upgraded_to_grammar():
    """A model draft that reproduces the FSM's forced string IS grammar.

    The FSM is deterministic for "ab" at the current position. The drafter
    happened to propose 'a','b','c'. The first two are grammar-forced at
    exactly those positions, so they can be flagged GRAMMAR and force-accepted;
    only 'c' is a genuine model guess.
    """
    fsm = MockFSM(transitions={(): "ab"}, valid_tokens={97, 98, 99})
    grammar = MockGrammar(fsm)

    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=MockTokenizer(),
        existing_model_draft=[97, 98, 99],
        budget=5,
    )

    assert plan.tokens == [97, 98, 99, -1, -1]
    assert plan.flags == [
        DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_MODEL,
        DRAFT_FLAG_EMPTY, DRAFT_FLAG_EMPTY
    ]


def test_model_draft_diverging_from_grammar_prefix_stays_model():
    """No upgrade when the drafter did not reproduce the forced string."""
    fsm = MockFSM(transitions={(): "ab"}, valid_tokens={97, 98, 99})
    grammar = MockGrammar(fsm)

    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=MockTokenizer(),
        existing_model_draft=[99, 97],
        budget=4,
    )

    assert plan.tokens[:2] == [99, 97]
    assert plan.flags[:2] == [DRAFT_FLAG_MODEL, DRAFT_FLAG_MODEL]


def test_existing_model_flags_are_preserved():
    """Provenance already established upstream must survive the round-trip.

    ``_topup_scheduled_spec_decode_tokens`` re-feeds last step's plan as
    ``existing_model_draft``. Without carrying the flags, a token that was
    proven grammar-forced gets demoted to MODEL and loses forced acceptance.
    """
    plan = CompositeDraftPlanner.plan(
        grammar=None,
        tokenizer=None,
        existing_model_draft=[10, 11],
        existing_model_flags=[DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_MODEL],
        budget=4,
    )

    assert plan.tokens == [10, 11, -1, -1]
    assert plan.flags == [
        DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_MODEL, DRAFT_FLAG_EMPTY,
        DRAFT_FLAG_EMPTY
    ]


def test_plan_output_always_satisfies_row_invariant():
    """Whatever phases fire, the emitted row must be legal for the kernel."""
    fsm = MockFSM(transitions={
        (): "ab",
        (97, 98, 99): "z"
    },
                  valid_tokens={97, 98, 99, 122})
    grammar = MockGrammar(fsm)
    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=MockTokenizer(),
        existing_model_draft=[97, 98, 99],
        budget=6,
    )
    assert validate_draft_row_invariant(plan.flags) is True
    assert len(plan.flags) == len(plan.tokens)


# ---------------------------------------------------------------------------
# Defect 4: dead code
# ---------------------------------------------------------------------------


def test_dead_topup_helper_is_gone():
    """``_topup_draft_token_ids`` had zero callers and a padding convention
    that contradicted ``_topup_scheduled_spec_decode_tokens``."""
    assert not hasattr(grammar_jump, "_topup_draft_token_ids")
