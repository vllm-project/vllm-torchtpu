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
"""Unit tests for the GaaD (tokens, flags) protocol and RejectionSampler integration.

Tests:
1. Flag constants and ordered row invariant (G^a D^b 0^c).
2. Grammar jump generation with slot flags.
3. RejectionSampler greedy path with forced acceptance (flag == 2).
4. RejectionSampler causal cascade truncation on model mismatch.
5. RejectionSampler random/stochastic path with flags.
6. Backward compatibility when draft_flags is None.
"""

from unittest.mock import Mock

import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID, RejectionSampler,
    _greedy_rejection_sample_with_segment)
from vllm_torchtpu.spec_decode.grammar_jump import (
    DRAFT_FLAG_EMPTY, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_MODEL,
    propose_jump_topup_with_flags, validate_draft_row_invariant)


class FakeMatcher:
    """Mock XGrammar matcher."""

    def __init__(self, jump_string=""):
        self._jump_string = jump_string
        self.accepted_tokens = []

    def accept_token(self, token_id: int) -> bool:
        self.accepted_tokens.append(token_id)
        return True

    def find_jump_forward_string(self) -> str:
        return self._jump_string

    def rollback(self, num_tokens: int) -> None:
        self.accepted_tokens = self.accepted_tokens[:-num_tokens]


class FakeTokenizer:
    """Mock Tokenizer converting chars to IDs."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def test_flag_constants_and_draft_row_invariant():
    """Verify flag definitions and the empty-suffix row invariant."""
    assert DRAFT_FLAG_EMPTY == 0
    assert DRAFT_FLAG_MODEL == 1
    assert DRAFT_FLAG_GRAMMAR == 2

    # Any interleaving of grammar/model is legal; only the empty tail matters.
    assert validate_draft_row_invariant([2, 2, 1, 1, 0, 0]) is True
    assert validate_draft_row_invariant([2, 2, 2, 0, 0]) is True
    assert validate_draft_row_invariant([1, 1, 1, 0]) is True
    assert validate_draft_row_invariant([0, 0, 0]) is True
    assert validate_draft_row_invariant([1, 2, 0]) is True  # G after D is fine

    # Violations: a hole in the middle, or an out-of-range flag.
    assert validate_draft_row_invariant([0, 1, 0]) is False
    assert validate_draft_row_invariant([2, 0, 1]) is False
    assert validate_draft_row_invariant([3, 0]) is False


def test_propose_jump_topup_with_flags():
    """Verify that propose_jump_topup_with_flags assigns correct flags."""
    grammar = Mock()
    grammar.is_terminated.return_value = False
    grammar.matcher = FakeMatcher(jump_string="xyz")
    tokenizer = FakeTokenizer()

    # Pre-existing model draft: [101, 102]
    validated_draft = [101, 102]
    num_spec_tokens = 6

    tokens, flags = propose_jump_topup_with_flags(
        grammar=grammar,
        tokenizer=tokenizer,
        validated_draft=validated_draft,
        num_spec_tokens=num_spec_tokens,
    )

    # "xyz" -> [ord('x'), ord('y'), ord('z')]
    expected_jump_tokens = [ord("x"), ord("y"), ord("z")]
    assert tokens == expected_jump_tokens
    # Each jump token must carry DRAFT_FLAG_GRAMMAR (2)
    assert flags == [DRAFT_FLAG_GRAMMAR] * len(expected_jump_tokens)


def test_greedy_rejection_sample_forced_acceptance_with_flags(device):
    """Verify that tokens with flag == 2 bypass argmax mismatch and are forced accepted."""
    # 1 sequence with 3 draft tokens: [10, 20, 30]
    draft_token_ids = torch.tensor([10, 20, 30],
                                   dtype=torch.int32,
                                   device=device)
    num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([40], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    max_draft_tokens = 3

    # Target model output:
    # Position 0: argmax is 99 (MISMATCH with draft token 10!)
    # Position 1: argmax is 20 (matches draft token 20)
    # Position 2: argmax is 30 (matches draft token 30)
    # Target logits has 1 row per draft token: shape (3, 128)
    target_logits = torch.zeros((3, 128), dtype=torch.float32, device=device)
    target_logits[0, 99] = 10.0
    target_logits[1, 20] = 10.0
    target_logits[2, 30] = 10.0

    # CASE 1: Standard verification (flags = [1, 1, 1], model draft)
    flags_model = torch.tensor([1, 1, 1], dtype=torch.int32, device=device)
    res_model = _greedy_rejection_sample_with_segment(
        draft_token_ids=draft_token_ids,
        target_logits=target_logits,
        num_draft_tokens=num_draft_tokens,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=max_draft_tokens,
        accept_u=None,
        synthetic_conditional_rates=None,
        synthetic_mode=False,
        draft_flags=flags_model,
    )
    # Token 0 mismatched (10 vs 99), so sequence immediately rejected at slot 0.
    # Output is target model's corrected token at slot 0: [99, PLACEHOLDER, ...]
    assert res_model[0, 0].item() == 99
    assert res_model[0, 1].item() == PLACEHOLDER_TOKEN_ID

    # CASE 2: Forced Acceptance (flags = [2, 1, 1], token 0 is grammar jump)
    flags_grammar = torch.tensor([2, 1, 1], dtype=torch.int32, device=device)
    res_grammar = _greedy_rejection_sample_with_segment(
        draft_token_ids=draft_token_ids,
        target_logits=target_logits,
        num_draft_tokens=num_draft_tokens,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=max_draft_tokens,
        accept_u=None,
        synthetic_conditional_rates=None,
        synthetic_mode=False,
        draft_flags=flags_grammar,
    )
    # Token 0 is forced accepted (flag=2), tokens 1 & 2 match target argmax, bonus is 40!
    # Expected accepted tokens: [10, 20, 30, 40]
    assert res_grammar[0, 0].item() == 10
    assert res_grammar[0, 1].item() == 20
    assert res_grammar[0, 2].item() == 30
    assert res_grammar[0, 3].item() == 40


def test_greedy_rejection_sample_causal_truncation(device):
    """Verify that when a model token (flag=1) rejects, subsequent tokens are truncated."""
    # Valid ordered row invariant (G^1 D^2): [10 (G), 20 (D), 30 (D)]
    raw_flags = [2, 1, 1]
    assert validate_draft_row_invariant(raw_flags) is True

    draft_token_ids = torch.tensor([10, 20, 30],
                                   dtype=torch.int32,
                                   device=device)
    num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([40], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    max_draft_tokens = 3

    # Target model output:
    # Slot 0: argmax 99 (G forced accept -> keeps 10)
    # Slot 1: argmax 88 (D mismatch with 20! -> rejected, recovers 88)
    # Slot 2: argmax 30 (D matches 30, but must be truncated due to slot 1 mismatch!)
    target_logits = torch.zeros((3, 128), dtype=torch.float32, device=device)
    target_logits[0, 99] = 10.0
    target_logits[1, 88] = 10.0
    target_logits[2, 30] = 10.0

    flags = torch.tensor(raw_flags, dtype=torch.int32, device=device)
    res = _greedy_rejection_sample_with_segment(
        draft_token_ids=draft_token_ids,
        target_logits=target_logits,
        num_draft_tokens=num_draft_tokens,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=max_draft_tokens,
        accept_u=None,
        synthetic_conditional_rates=None,
        synthetic_mode=False,
        draft_flags=flags,
    )
    # Slot 0 accepted (10), Slot 1 rejected -> recovery token 88. Slot 2 truncated!
    assert res[0, 0].item() == 10
    assert res[0, 1].item() == 88
    assert res[0, 2].item() == PLACEHOLDER_TOKEN_ID
    assert res[0, 3].item() == PLACEHOLDER_TOKEN_ID


def test_random_rejection_sample_forced_acceptance_with_flags(device):
    """Verify forced acceptance (flag == 2) on the stochastic rejection path (do_sampling=True)."""
    sampler = RejectionSampler(device=device)

    # 2 draft tokens: [10, 20]
    draft_token_ids = torch.tensor([10, 20], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([30], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)

    # Target distribution:
    # Position 0: token 99 has huge logit (100.0), so p(10) ~ 0.0
    # Position 1: token 20 has huge logit (100.0), so p(20) ~ 1.0
    target_logits = torch.zeros((2, 128), dtype=torch.float32, device=device)
    target_logits[0, 99] = 100.0
    target_logits[1, 20] = 100.0

    temperatures = torch.ones((2, 1), dtype=torch.float32, device=device)
    top_k = torch.zeros((2, 1), dtype=torch.int32, device=device)
    top_p = torch.ones((2, 1), dtype=torch.float32, device=device)
    accept_u = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
    recover_u = torch.full_like(target_logits, 0.5)

    # CASE 1: Standard model draft flags [1, 1] -> Position 0 rejects because p(10) < 0.5
    flags_model = torch.tensor([1, 1], dtype=torch.int32, device=device)
    out_model = sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=2,
        temperatures=temperatures,
        top_k=top_k,
        top_p=top_p,
        accept_u=accept_u,
        recover_u=recover_u,
        do_sampling=True,
        draft_flags=flags_model,
    )
    assert out_model[0, 0].item() == 99
    assert out_model[0, 1].item() == PLACEHOLDER_TOKEN_ID

    # CASE 2: Grammar jump flag at slot 0: [2, 1] -> Position 0 forced accepted!
    flags_grammar = torch.tensor([2, 1], dtype=torch.int32, device=device)
    out_grammar = sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=2,
        temperatures=temperatures,
        top_k=top_k,
        top_p=top_p,
        accept_u=accept_u,
        recover_u=recover_u,
        do_sampling=True,
        draft_flags=flags_grammar,
    )
    assert out_grammar[0, 0].item() == 10
    assert out_grammar[0, 1].item() == 20
    assert out_grammar[0, 2].item() == 30


def test_rejection_sampler_class_interface_backward_compatible(device):
    """Verify RejectionSampler class works with draft_flags=None (backward compatibility)."""
    sampler = RejectionSampler(device=device)

    draft_token_ids = torch.tensor([1, 2], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([3], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)

    target_logits = torch.zeros((2, 10), dtype=torch.float32, device=device)
    target_logits[0, 1] = 10.0
    target_logits[1, 2] = 10.0

    # draft_flags omitted (defaults to None)
    out = sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=2,
    )
    assert out[0, 0].item() == 1
    assert out[0, 1].item() == 2
    assert out[0, 2].item() == 3


def test_grammar_flag_never_force_accepts_padding_slot(device):
    """A GRAMMAR flag must never resurrect a ``-1`` padding slot.

    ``draft_token_ids`` is padded with -1 (``PLACEHOLDER_TOKEN_ID``) up to the
    static budget. If a stale or mis-indexed flag tensor puts a 2 on a padding
    slot, unguarded forced acceptance would emit -1 as a real output token and
    keep the sequence alive past its true end. The acceptance mask must be
    guarded by ``draft_token_ids >= 0``.
    """
    # Slot 0 is padding (-1) but adversarially flagged GRAMMAR.
    draft_token_ids = torch.tensor([-1, 20], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([30], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)

    target_logits = torch.zeros((2, 128), dtype=torch.float32, device=device)
    target_logits[0, 99] = 10.0
    target_logits[1, 20] = 10.0

    flags = torch.tensor([2, 1], dtype=torch.int32, device=device)
    res = _greedy_rejection_sample_with_segment(
        draft_token_ids=draft_token_ids,
        target_logits=target_logits,
        num_draft_tokens=num_draft_tokens,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=2,
        accept_u=None,
        synthetic_conditional_rates=None,
        synthetic_mode=False,
        draft_flags=flags,
    )
    # The padding slot must be rejected and recovered from the target argmax,
    # which truncates everything after it.
    assert res[0, 0].item() == 99
    assert res[0, 1].item() == PLACEHOLDER_TOKEN_ID


def test_all_zero_draft_flags_behaves_like_none(device):
    """An all-zero flag tensor must be a no-op.

    This is a characterization guard, not a regression test: the rejection
    sampler is ``torch.compile``d with a Python-level ``if draft_flags is not
    None``, so Dynamo emits two graphs. ``_precompile_rejection_sampler`` warms
    the flag-carrying graph with a zeros tensor, which is only safe if zeros are
    observationally identical to ``None``.
    """
    sampler = RejectionSampler(device=device)

    draft_token_ids = torch.tensor([1, 2], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([3], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)

    target_logits = torch.zeros((2, 10), dtype=torch.float32, device=device)
    target_logits[0, 1] = 10.0
    target_logits[1, 2] = 10.0

    common = dict(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=2,
    )
    out_none = sampler(**common)
    out_zeros = sampler(
        **common,
        draft_flags=torch.zeros(2, dtype=torch.int32, device=device),
    )
    assert out_none.cpu().tolist() == out_zeros.cpu().tolist()
