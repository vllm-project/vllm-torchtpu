# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests the grammar jump-decoding top-up without loading the TPU runtime.

The module under test is loaded by path, following
``tests/platforms/test_patch_registry.py``: importing ``vllm_torchtpu`` pulls in
JAX and TorchTPU, which the top-up core deliberately does not need.

The fake matcher records an ordered operation log and keys its jump strings by
the *current* FSM position. That is what makes the ordering requirement
testable: a top-up that peeks before walking past the proposer's draft reads the
jump string of the wrong state, and no assertion on the returned tokens alone
would catch it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (Path(__file__).resolve().parents[2] /
                "src/vllm_torchtpu/spec_decode/grammar_jump.py")


def _load_module():
    name = "_test_grammar_jump"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    module = _load_module()
    yield module
    sys.modules.pop(module.__name__, None)


class FakeMatcher:
    """Minimal ``xgr.GrammarMatcher`` stand-in with an operation log.

    ``jump_by_state`` maps a tuple of currently-accepted tokens to the jump
    string the FSM would report at that position.
    """

    def __init__(self, jump_by_state=None, rejects=(), max_rollback=8):
        self.jump_by_state = jump_by_state or {}
        self.rejects = set(rejects)
        self.max_rollback = max_rollback
        self.accepted: list[int] = []
        self.log: list[tuple] = []
        self.terminated = False

    def accept_token(self, token: int) -> bool:
        self.log.append(("accept", token))
        if token in self.rejects:
            return False
        self.accepted.append(token)
        return True

    def rollback(self, num_tokens: int) -> None:
        self.log.append(("rollback", num_tokens))
        if num_tokens > self.max_rollback:
            raise AssertionError(
                f"rollback({num_tokens}) exceeds max_rollback_tokens="
                f"{self.max_rollback}; xgrammar would raise here")
        if num_tokens > len(self.accepted):
            raise AssertionError(
                f"rollback({num_tokens}) exceeds accepted depth "
                f"{len(self.accepted)}")
        if num_tokens:
            del self.accepted[-num_tokens:]

    def find_jump_forward_string(self) -> str:
        state = tuple(self.accepted)
        self.log.append(("find", state))
        return self.jump_by_state.get(state, "")

    def is_terminated(self) -> bool:
        return self.terminated


class FakeGrammar:

    def __init__(self, matcher, terminated=False):
        self.matcher = matcher
        self._terminated = terminated

    def is_terminated(self) -> bool:
        return self._terminated


class FakeTokenizer:
    """Encodes a string as one token per character, using code points as ids."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in self.mapping:
            return list(self.mapping[text])
        return [ord(c) for c in text]


def test_peeks_from_the_state_after_the_proposers_draft(mod):
    """The jump string must be read *after* walking past the draft.

    A top-up that peeks from the base state would return the tokens of
    "WRONG" and splice them in after the draft, proposing a sequence the
    grammar never allows.
    """
    matcher = FakeMatcher(jump_by_state={
        (): "WRONG",
        (10, 11): "ok",
    })
    grammar = FakeGrammar(matcher)
    tokenizer = FakeTokenizer()

    result = mod.propose_jump_topup(grammar,
                                    tokenizer,
                                    validated_draft=[10, 11],
                                    num_spec_tokens=8)

    assert result == [ord("o"), ord("k")]
    # The draft tokens were accepted before the FSM was queried.
    find_index = matcher.log.index(("find", (10, 11)))
    assert matcher.log[:find_index] == [("accept", 10), ("accept", 11)]


def test_restores_matcher_state_exactly(mod):
    matcher = FakeMatcher(jump_by_state={(10, ): "abc"})
    grammar = FakeGrammar(matcher)

    result = mod.propose_jump_topup(grammar,
                                    FakeTokenizer(),
                                    validated_draft=[10],
                                    num_spec_tokens=8)

    assert result
    assert matcher.accepted == [], "FSM must be left exactly as it was found"


def test_budget_is_num_spec_tokens_minus_draft(mod):
    matcher = FakeMatcher(jump_by_state={(1, 2): "abcdef"})
    grammar = FakeGrammar(matcher)

    result = mod.propose_jump_topup(grammar,
                                    FakeTokenizer(),
                                    validated_draft=[1, 2],
                                    num_spec_tokens=5)

    assert len(result) == 3, "5 slots minus a 2-token draft leaves 3"


def test_rollback_never_exceeds_max_rollback_tokens(mod):
    """xgrammar caps rollback at ``num_speculative_tokens``.

    The total walk is draft + jump, so the budget has to be derived from
    ``num_spec_tokens`` rather than taken from the caller, or the restore at
    the end raises.
    """
    matcher = FakeMatcher(jump_by_state={(1, 2, 3): "abcdefghij"},
                          max_rollback=4)
    grammar = FakeGrammar(matcher)

    mod.propose_jump_topup(grammar,
                           FakeTokenizer(),
                           validated_draft=[1, 2, 3],
                           num_spec_tokens=4)

    rollbacks = [n for op, n in matcher.log if op == "rollback"]
    assert rollbacks == [4]


def test_full_draft_leaves_no_budget(mod):
    matcher = FakeMatcher(jump_by_state={(1, 2): "abc"})
    grammar = FakeGrammar(matcher)

    result = mod.propose_jump_topup(grammar,
                                    FakeTokenizer(),
                                    validated_draft=[1, 2],
                                    num_spec_tokens=2)

    assert result == []
    assert matcher.log == [], "no budget means the FSM is never touched"


def test_terminated_grammar_returns_empty(mod):
    matcher = FakeMatcher(jump_by_state={(): "abc"})
    grammar = FakeGrammar(matcher, terminated=True)

    result = mod.propose_jump_topup(grammar,
                                    FakeTokenizer(),
                                    validated_draft=[],
                                    num_spec_tokens=8)

    assert result == []
    assert matcher.log == []


def test_no_jump_string_restores_state(mod):
    matcher = FakeMatcher(jump_by_state={})
    grammar = FakeGrammar(matcher)

    result = mod.propose_jump_topup(grammar,
                                    FakeTokenizer(),
                                    validated_draft=[7],
                                    num_spec_tokens=8)

    assert result == []
    assert matcher.accepted == []
    assert ("rollback", 1) in matcher.log


def test_tokenizer_boundary_mismatch_keeps_accepted_prefix(mod):
    """``encode(jump_str)`` can disagree with what the FSM accepts.

    Only the prefix the FSM actually accepted may be proposed; the rest is
    dropped and the state is still restored in full.
    """
    matcher = FakeMatcher(jump_by_state={(): "abc"}, rejects=[ord("b")])
    grammar = FakeGrammar(matcher)
    stats = mod.JumpStats()

    result = mod.propose_jump_topup(grammar,
                                    FakeTokenizer(),
                                    validated_draft=[],
                                    num_spec_tokens=8,
                                    stats=stats)

    assert result == [ord("a")]
    assert matcher.accepted == []
    assert stats.fsm_rejections == 1


def test_stats_track_opportunity_and_yield(mod):
    matcher = FakeMatcher(jump_by_state={(): "abcdef"})
    grammar = FakeGrammar(matcher)
    stats = mod.JumpStats()

    mod.propose_jump_topup(grammar,
                           FakeTokenizer(),
                           validated_draft=[],
                           num_spec_tokens=4,
                           stats=stats)
    # Second call: no jump available.
    mod.propose_jump_topup(FakeGrammar(FakeMatcher()),
                           FakeTokenizer(),
                           validated_draft=[],
                           num_spec_tokens=4,
                           stats=stats)

    assert stats.calls == 2
    assert stats.hits == 1
    assert stats.tokens_proposed == 4
    assert stats.truncations == 1


def test_empty_jump_encoding_is_not_proposed(mod):
    matcher = FakeMatcher(jump_by_state={(): "zz"})
    grammar = FakeGrammar(matcher)
    tokenizer = FakeTokenizer(mapping={"zz": []})

    result = mod.propose_jump_topup(grammar,
                                    tokenizer,
                                    validated_draft=[],
                                    num_spec_tokens=8)

    assert result == []


def test_unacceptable_draft_aborts_and_restores(mod):
    """A draft the FSM rejects mid-way must not leave the matcher advanced."""
    matcher = FakeMatcher(jump_by_state={(1, ): "abc"}, rejects=[2])
    grammar = FakeGrammar(matcher)

    result = mod.propose_jump_topup(grammar,
                                    FakeTokenizer(),
                                    validated_draft=[1, 2],
                                    num_spec_tokens=8)

    assert result == []
    assert matcher.accepted == []
