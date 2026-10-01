# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GaaD module: constants, invariants, FSMBridge, and GaadProposer."""

from unittest.mock import MagicMock

import torch

from vllm_torchtpu.spec_decode.gaad import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    DRAFT_FLAG_MODEL,
    FSMBridge,
    GaadProposer,
    validate_draft_row_invariant,
)


class FakeMatcher:
    """Mock XGrammar matcher for deterministic traversal."""

    def __init__(self, jump_string="", accept_map=None):
        self._jump_string = jump_string
        self._accept_map = accept_map or {}
        self.accepted_tokens = []
        self.rollback_calls = []

    def accept_token(self, token_id: int) -> bool:
        can_accept = self._accept_map.get(token_id, True)
        if can_accept:
            self.accepted_tokens.append(token_id)
        return can_accept

    def find_jump_forward_string(self) -> str:
        return self._jump_string

    def rollback(self, count: int) -> None:
        self.rollback_calls.append(count)
        del self.accepted_tokens[-count:]


class FakeGrammar:
    """Mock StructuredOutputGrammar."""

    def __init__(self, matcher, is_terminated=False):
        self.matcher = matcher
        self._is_terminated = is_terminated

    def is_terminated(self) -> bool:
        return self._is_terminated


class FakeTokenizer:
    """Mock tokenizer for character-level or sub-word mapping."""

    def __init__(self, vocab_map=None):
        self.vocab_map = vocab_map or {
            ": ": [10, 20],
            '{"name": "': [100, 101, 102],
            "invalid": [999],
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.vocab_map.get(text, [ord(c) for c in text])


class TestGaadConstantsAndRowInvariant:
    """Tests for flag constants and validate_draft_row_invariant."""

    def test_constants_values(self):
        assert DRAFT_FLAG_EMPTY == 0
        assert DRAFT_FLAG_MODEL == 1
        assert DRAFT_FLAG_GRAMMAR == 2

    def test_valid_draft_rows(self):
        # Empty rows or pure padding
        assert validate_draft_row_invariant([])
        assert validate_draft_row_invariant([0, 0, 0, 0])

        # Pure grammar jumps
        assert validate_draft_row_invariant([2, 2, 2, 0])
        assert validate_draft_row_invariant([2, 2, 2, 2])

        # Pure model drafts
        assert validate_draft_row_invariant([1, 1, 0, 0])

        # Interleaved grammar and model drafts, ending with zeros
        assert validate_draft_row_invariant([2, 2, 1, 2, 0])
        assert validate_draft_row_invariant([1, 2, 1, 2, 2])

    def test_invalid_draft_rows_internal_holes(self):
        # Internal zeros (padding hole before valid draft) are forbidden
        assert not validate_draft_row_invariant([2, 0, 2, 0])
        assert not validate_draft_row_invariant([0, 1, 0, 0])
        assert not validate_draft_row_invariant([1, 0, 2, 0])

    def test_invalid_draft_rows_illegal_flags(self):
        # Flags outside {0, 1, 2}
        assert not validate_draft_row_invariant([3, 0, 0])
        assert not validate_draft_row_invariant([-1, 0, 0])
        assert not validate_draft_row_invariant([2, 4, 0])


class TestFSMBridge:
    """Tests for FSMBridge deterministic jump extraction and token
    boundary validation."""

    def test_extract_deterministic_jump_clean(self):
        bridge = FSMBridge()
        matcher = FakeMatcher(jump_string='{"name": "')
        grammar = FakeGrammar(matcher)
        tokenizer = FakeTokenizer()

        tokens, flags = bridge.extract_deterministic_jump(
            grammar=grammar,
            tokenizer=tokenizer,
            validated_draft=[],
            num_spec_tokens=5,
        )

        assert tokens == [100, 101, 102]
        assert flags == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR]
        # Matcher must be rolled back completely to preserve peek semantics
        assert len(matcher.accepted_tokens) == 0
        assert matcher.rollback_calls == [3]

    def test_extract_deterministic_jump_budget_cap(self):
        bridge = FSMBridge()
        matcher = FakeMatcher(jump_string='{"name": "')
        grammar = FakeGrammar(matcher)
        tokenizer = FakeTokenizer()

        tokens, flags = bridge.extract_deterministic_jump(
            grammar=grammar,
            tokenizer=tokenizer,
            validated_draft=[],
            num_spec_tokens=2,  # budget capped to 2
        )

        assert tokens == [100, 101]
        assert flags == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR]
        assert len(matcher.accepted_tokens) == 0
        assert matcher.rollback_calls == [2]

    def test_extract_deterministic_jump_boundary_rejection(self):
        bridge = FSMBridge()
        # Token 102 rejected by grammar
        matcher = FakeMatcher(
            jump_string='{"name": "', accept_map={100: True, 101: True, 102: False}
        )
        grammar = FakeGrammar(matcher)
        tokenizer = FakeTokenizer()

        tokens, flags = bridge.extract_deterministic_jump(
            grammar=grammar,
            tokenizer=tokenizer,
            validated_draft=[],
            num_spec_tokens=5,
        )

        assert tokens == [100, 101]
        assert flags == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR]
        assert len(matcher.accepted_tokens) == 0
        assert matcher.rollback_calls == [2]

    def test_extract_deterministic_jump_terminated_grammar(self):
        bridge = FSMBridge()
        matcher = FakeMatcher(jump_string='{"name": "')
        grammar = FakeGrammar(matcher, is_terminated=True)
        tokenizer = FakeTokenizer()

        tokens, flags = bridge.extract_deterministic_jump(
            grammar=grammar,
            tokenizer=tokenizer,
            validated_draft=[],
            num_spec_tokens=5,
        )

        assert tokens == []
        assert flags == []


class TestGaadProposer:
    """Tests for GaadProposer batch proposal interface."""

    def test_propose_batch_mixed_requests(self):
        vllm_config = MagicMock()
        vllm_config.speculative_config.num_speculative_tokens = 4
        proposer = GaadProposer(vllm_config)

        matcher1 = FakeMatcher(jump_string=": ")
        grammar1 = FakeGrammar(matcher1)

        req_structured = MagicMock()
        req_structured.is_finished.return_value = False
        req_structured.is_prefill_chunk = False
        req_structured.output_token_ids = [1]
        req_structured.max_tokens = 100
        req_structured.structured_output_request.grammar = grammar1

        req_unconstrained = MagicMock()
        req_unconstrained.is_finished.return_value = False
        req_unconstrained.is_prefill_chunk = False
        req_unconstrained.output_token_ids = [1]
        req_unconstrained.max_tokens = 100
        req_unconstrained.structured_output_request = None

        tokenizer = FakeTokenizer()

        batch_proposals = proposer.propose(
            requests=[req_structured, req_unconstrained],
            tokenizer=tokenizer,
        )

        assert len(batch_proposals) == 2
        # First request emitted grammar jump tokens [10, 20]
        assert batch_proposals[0] == [10, 20]
        # Second unconstrained request emitted empty list
        assert batch_proposals[1] == []


class TestGaadRuntimeFlagsAndRejectionSampler:
    """Tests for RejectionSampler forced acceptance and zero-target-logits bypass."""

    def test_rejection_sampler_forced_acceptance_flag2(self):
        """Tokens with flag == 2 are forced accepted even on argmax mismatch."""
        from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
            RejectionSampler,
        )

        sampler = RejectionSampler()
        device = torch.device("cpu")

        # 1 request with 2 draft tokens
        num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
        vocab_size = 50
        # target logits predict token 0 at step 0, and token 0 at step 1
        target_logits = torch.zeros((2, vocab_size), dtype=torch.float32, device=device)
        target_logits[0, 0] = 10.0
        target_logits[1, 0] = 10.0

        bonus_token_ids = torch.tensor([42], dtype=torch.int32, device=device)
        segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
        group_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)

        # Draft tokens are 10, 20 (mismatch with target argmax 0, 0)
        draft_token_ids = torch.tensor([10, 20], dtype=torch.int32, device=device)

        # Both tokens are GRAMMAR (flag == 2) -> forced accept!
        draft_flags = torch.tensor(
            [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR],
            dtype=torch.int32,
            device=device,
        )

        output = sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            segment_ids=segment_ids,
            group_indices=group_indices,
            max_draft_tokens=2,
            draft_flags=draft_flags,
        )

        # Both draft tokens forced accepted plus bonus token [10, 20, 42]
        assert output[0].tolist() == [10, 20, 42]

    def test_rejection_sampler_causal_cascade_truncation(self):
        """Model mismatch at slot 0 truncates subsequent grammar tokens at slot 1."""
        from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
            RejectionSampler,
        )

        sampler = RejectionSampler()
        device = torch.device("cpu")

        num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
        vocab_size = 50
        target_logits = torch.zeros((2, vocab_size), dtype=torch.float32, device=device)
        # Target argmax is 99 (say 5) at slot 0, and 20 at slot 1
        target_logits[0, 5] = 10.0
        target_logits[1, 20] = 10.0

        bonus_token_ids = torch.tensor([42], dtype=torch.int32, device=device)
        segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
        group_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)

        # Draft is [10, 20]. Slot 0 has flag MODEL (1), Slot 1 has flag GRAMMAR (2)
        draft_token_ids = torch.tensor([10, 20], dtype=torch.int32, device=device)
        draft_flags = torch.tensor(
            [DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR],
            dtype=torch.int32,
            device=device,
        )

        output = sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            segment_ids=segment_ids,
            group_indices=group_indices,
            max_draft_tokens=2,
            draft_flags=draft_flags,
        )

        # Slot 0 rejected -> recovers target argmax (5),
        # slot 1 truncated to PLACEHOLDER (-1)
        assert output[0, 0].item() == 5
        assert output[0, 1].item() == -1

    def test_rejection_sampler_zero_target_logits_bypass(self):
        """When target_logits is None and all drafts are GRAMMAR,
        execute zero-logits bypass."""
        from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
            RejectionSampler,
        )

        sampler = RejectionSampler()
        device = torch.device("cpu")

        num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
        bonus_token_ids = torch.tensor([42], dtype=torch.int32, device=device)
        segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
        group_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        draft_token_ids = torch.tensor([10, 20], dtype=torch.int32, device=device)
        draft_flags = torch.tensor(
            [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR],
            dtype=torch.int32,
            device=device,
        )

        output = sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            target_logits=None,  # Zero-Target-Logits Bypass!
            bonus_token_ids=bonus_token_ids,
            segment_ids=segment_ids,
            group_indices=group_indices,
            max_draft_tokens=2,
            draft_flags=draft_flags,
        )

        assert output[0].tolist() == [10, 20, 42]
