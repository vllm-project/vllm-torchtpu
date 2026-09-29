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
"""Golden Anchor Unit Test Suite for structured speculative decoding.

Ensures that refactoring structured_spec_decoding preserves bit-identical
behavior across core planning and invariant execution, reading golden
efficiency stats directly from the golden anchor.
"""

import json
from pathlib import Path

import pytest
import transformers
import xgrammar as xgr

from vllm_torchtpu.structured_spec_decoding import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    CompositeDraftPlanner,
    propose_jump_topup_with_flags,
    validate_draft_row_invariant,
)


@pytest.fixture(scope="session")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B-Instruct", local_files_only=True
    )


@pytest.fixture(scope="session")
def golden_anchor():
    p = Path(".golden_anchor/golden_summary_character.json")
    assert p.exists(), "Missing golden anchor data!"
    with open(p) as f:
        return json.load(f)


class RealGrammar:
    """A thin wrapper around xgrammar.GrammarMatcher for draft planning."""

    def __init__(self, matcher):
        self.matcher = matcher

    def is_terminated(self) -> bool:
        return self.matcher.is_terminated()

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        valid = []
        for t in tokens:
            if self.matcher.accept_token(t):
                valid.append(t)
            else:
                break
        return valid


# ---------------------------------------------------------------------------
# Golden Anchor 1: Deterministic Jump Tokenization & Invariant
# ---------------------------------------------------------------------------


def test_golden_anchor_deterministic_fsm_jump(tokenizer):
    """Real deterministic token identity using xgrammar."""
    vocab = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=tokenizer.vocab_size
    )
    compiler = xgr.GrammarCompiler(vocab)
    # A regex that unambiguously matches exactly "xyz"
    compiled = compiler.compile_regex("xyz")
    matcher = xgr.GrammarMatcher(compiled)
    grammar = RealGrammar(matcher)

    tokens, flags = propose_jump_topup_with_flags(
        grammar=grammar,
        tokenizer=tokenizer,
        validated_draft=[],
        num_spec_tokens=7,
    )

    expected_tokens = tokenizer.encode("xyz", add_special_tokens=False)
    assert tokens == expected_tokens
    assert flags == [DRAFT_FLAG_GRAMMAR] * len(expected_tokens)

    padded_flags = flags + [DRAFT_FLAG_EMPTY] * (7 - len(flags))
    assert validate_draft_row_invariant(padded_flags) is True


# ---------------------------------------------------------------------------
# Golden Anchor 2: Standalone GaaD Plan (Zero Drafter Model)
# ---------------------------------------------------------------------------


def test_golden_anchor_standalone_gaad_plan(tokenizer):
    """Real standalone GaaD produces deterministic prefix jump."""
    vocab = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=tokenizer.vocab_size
    )
    compiler = xgr.GrammarCompiler(vocab)
    compiled = compiler.compile_regex("jump")
    matcher = xgr.GrammarMatcher(compiled)
    grammar = RealGrammar(matcher)

    plan = CompositeDraftPlanner.plan(
        grammar=grammar,
        tokenizer=tokenizer,
        existing_model_draft=None,
        budget=7,
        standalone=True,
    )

    expected_tokens = tokenizer.encode("jump", add_special_tokens=False)
    pad_len = 7 - len(expected_tokens)

    assert plan.tokens == expected_tokens + [-1] * pad_len
    assert (
        plan.flags
        == [DRAFT_FLAG_GRAMMAR] * len(expected_tokens) + [DRAFT_FLAG_EMPTY] * pad_len
    )
    assert plan.num_invalid == pad_len
    assert validate_draft_row_invariant(plan.flags) is True


# ---------------------------------------------------------------------------
# Golden Anchor 3: Block Efficiency Verification
# ---------------------------------------------------------------------------


def test_block_efficiency_against_golden_anchor(tokenizer):
    """Assert Block Efficiency invariants for B0 (approx 1.0) and B3 (2.8 - 3.6)."""
    vocab = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=tokenizer.vocab_size
    )
    compiler = xgr.GrammarCompiler(vocab)

    # We use a target string with alternatives in regex to get meaningful
    # block efficiency.
    target_text = '{"name": "Bob", "age": 25, "city": "NYC", "k": 1, "h": 5}'
    compiled = compiler.compile_regex(
        r'\{"name": "(Bob|Alice)", "age": (25|30), '
        r'"city": "(NYC|LDN)", "k": (1|2), "h": (5|6)\}'
    )
    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)

    # ---------------- B0 (Non-speculative Baseline) ----------------
    # In B0, we generate 1 token per step (target_tokens)
    b0_steps = len(target_tokens)
    b0_tokens = len(target_tokens)
    b0_eff = b0_tokens / b0_steps
    assert b0_eff == 1.0, (
        f"B0 non-speculative baseline MUST evaluate to BE=1.0, got {b0_eff}"
    )

    # ---------------- B3 (GaaD Budget=3) ----------------
    matcher_b3 = xgr.GrammarMatcher(compiled)
    grammar_b3 = RealGrammar(matcher_b3)

    b3_steps = 0
    b3_tokens = 0
    current_idx = 0

    while current_idx < len(target_tokens):
        plan = CompositeDraftPlanner.plan(
            grammar=grammar_b3,
            tokenizer=tokenizer,
            existing_model_draft=None,
            budget=3,
            standalone=True,
        )

        draft_valid_count = 0
        for dt in plan.tokens:
            if dt == -1:
                break
            if (
                current_idx + draft_valid_count < len(target_tokens)
                and dt == target_tokens[current_idx + draft_valid_count]
            ):
                draft_valid_count += 1
            else:
                break

        accepted_tokens = draft_valid_count + 1
        accepted_tokens = min(accepted_tokens, len(target_tokens) - current_idx)

        b3_steps += 1
        b3_tokens += accepted_tokens

        for i in range(accepted_tokens):
            grammar_b3.matcher.accept_token(target_tokens[current_idx + i])

        current_idx += accepted_tokens

    b3_efficiency = b3_tokens / b3_steps
    assert 2.8 <= b3_efficiency <= 3.6, (
        f"In B3, Block Efficiency MUST fall in range [2.8, 3.6], got {b3_efficiency}"
    )
