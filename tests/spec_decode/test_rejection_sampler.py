# Copyright 2025 Google LLC
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

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch._dynamo

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID, RejectionSampler)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    MASKED_LOGIT_VALUE, apply_top_k_top_p)


@pytest.fixture(autouse=True)
def _raise_dynamo_recompile_limit():
    """The sampler is torch.compiled with dynamic=False, so every distinct
    input shape recompiles. Production raises this limit in the runner
    (tpu_runner.py); the unit test never builds a runner, so the default (8)
    is hit once the parametrized cases span >8 shapes. Mirror the runner here
    and restore afterwards so the bump doesn't leak into other tests."""
    cfg = torch._dynamo.config
    names = [
        n for n in ("recompile_limit", "cache_size_limit") if hasattr(cfg, n)
    ]
    saved = {n: getattr(cfg, n) for n in names}
    for n in names:
        setattr(cfg, n, 1024)
    yield
    for n, v in saved.items():
        setattr(cfg, n, v)


def test_rejection_sampler_greedy_all_accepted(device):
    sampler = RejectionSampler()
    num_draft_tokens = torch.tensor([3, 2], dtype=torch.int32, device=device)

    # Target logits should match the draft tokens at corresponding indices
    # To make argmax match, we set a high logit value at the draft index
    vocab_size = 500
    target_logits = torch.zeros((5, vocab_size),
                                dtype=torch.float32,
                                device=device)
    # First request tokens: index 0, 1, 2
    target_logits[0, 10] = 10.0
    target_logits[1, 20] = 10.0
    target_logits[2, 30] = 10.0
    # Second request tokens: index 3, 4
    target_logits[3, 100] = 10.0
    target_logits[4, 200] = 10.0

    bonus_token_ids = torch.tensor([40, 300], dtype=torch.int32, device=device)

    # Segment IDs & Group Indices (flattened representation)
    # First request: indices 0, 1, 2 in target_logits
    # Second request: indices 3, 4 in target_logits
    segment_ids = torch.tensor([0, 0, 0, 1, 1],
                               dtype=torch.int64,
                               device=device)
    group_indices = torch.tensor([0, 1, 2, 0, 1],
                                 dtype=torch.int64,
                                 device=device)

    # Flatten draft tokens for the segment forward call
    flat_draft_token_ids = torch.tensor([10, 20, 30, 100, 200],
                                        dtype=torch.int32,
                                        device=device)

    output = sampler(draft_token_ids=flat_draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=3)

    # Expected output shape: [batch_size, max_draft_tokens + 1] -> [2, 4]
    # Request 0: 3 draft tokens accepted (10, 20, 30) + bonus (40) -> [10, 20, 30, 40]
    # Request 1: 2 draft tokens accepted (100, 200) + bonus (300) + padding (-1) -> [100, 200, 300, -1]
    expected = torch.tensor([[10, 20, 30, 40], [100, 200, 300, -1]],
                            dtype=torch.int32,
                            device=device)

    assert torch.equal(output, expected)


def test_rejection_sampler_greedy_with_mismatches(device):
    sampler = RejectionSampler()

    # Batch size = 2, max_draft_tokens = 3
    # Request 0: mismatch at index 1 (draft has 20, target argmax will be 99)
    # Request 1: mismatch at index 0 (draft has 100, target argmax will be 888)
    flat_draft_token_ids = torch.tensor([10, 20, 30, 100, 200],
                                        dtype=torch.int32,
                                        device=device)
    num_draft_tokens = torch.tensor([3, 2], dtype=torch.int32, device=device)

    vocab_size = 1000
    target_logits = torch.zeros((5, vocab_size),
                                dtype=torch.float32,
                                device=device)
    # Request 0 target argmax: [10, 99, 30] (index 1 mismatched!)
    target_logits[0, 10] = 10.0
    target_logits[1, 99] = 10.0
    target_logits[2, 30] = 10.0
    # Request 1 target argmax: [888, 200] (index 0 mismatched!)
    target_logits[3, 888] = 10.0
    target_logits[4, 200] = 10.0

    bonus_token_ids = torch.tensor([40, 300], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0, 0, 1, 1],
                               dtype=torch.int64,
                               device=device)
    group_indices = torch.tensor([0, 1, 2, 0, 1],
                                 dtype=torch.int64,
                                 device=device)

    output = sampler(draft_token_ids=flat_draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=3)

    # Request 0: index 0 accepted (10), index 1 mismatched -> accept target argmax (99), subsequent masked.
    # -> [10, 99, -1, -1]
    # Request 1: index 0 mismatched -> accept target argmax (888), subsequent masked.
    # -> [888, -1, -1, -1]
    expected = torch.tensor([[10, 99, -1, -1], [888, -1, -1, -1]],
                            dtype=torch.int32,
                            device=device)

    assert torch.equal(output, expected)


def test_rejection_sampler_greedy_zero_draft_tokens(device):
    sampler = RejectionSampler()

    # Batch size = 2, max_draft_tokens = 3
    # Both requests have 0 draft tokens (prefill / fallback)
    flat_draft_token_ids = torch.tensor([], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([0, 0], dtype=torch.int32, device=device)

    target_logits = torch.zeros((0, 10), dtype=torch.float32,
                                device=device)  # No tokens mapped
    bonus_token_ids = torch.tensor([40, 300], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([], dtype=torch.int64, device=device)
    group_indices = torch.tensor([], dtype=torch.int64, device=device)

    output = sampler(draft_token_ids=flat_draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=3)

    # Zero draft tokens -> accept bonus tokens instantly at index 0
    # -> [40, -1, -1, -1]
    # -> [300, -1, -1, -1]
    expected = torch.tensor([[40, -1, -1, -1], [300, -1, -1, -1]],
                            dtype=torch.int32,
                            device=device)

    assert torch.equal(output, expected)


VOCAB_SIZE = 256


@dataclass
class RejectionSamplerCase:
    """A greedy rejection-sampling scenario.

    `target_tokens` are the per-position target argmax ids (length == total
    draft tokens). `expected` mirrors the sibling's variable-length rows; the
    test pads them to the torch sampler's fixed matrix width.
    """
    name: str
    draft_tokens: list[int]
    target_tokens: list[int]
    num_draft_per_seq: list[int]
    bonus_tokens: list[int]
    expected: list[list[int]]


# One representative case per distinct branch of the greedy sampler.
GREEDY_CASES: list[RejectionSamplerCase] = [
    # accept-all + bonus (single seq).
    RejectionSamplerCase("perfect_match", [1, 2, 3], [1, 2, 3], [3], [4],
                         [[1, 2, 3, 4]]),
    # mismatch mid-sequence -> corrected token, rest masked, no bonus.
    RejectionSamplerCase("early_mismatch", [1, 2, 3], [1, 5, 3], [3], [4],
                         [[1, 5]]),
    # mismatch at position 0.
    RejectionSamplerCase("first_token_mismatch", [1], [2], [1], [3], [[2]]),
    # multi-seq batch mixing accept-all and mismatch.
    RejectionSamplerCase("multiple_sequences", [1, 2, 3, 4], [1, 2, 3, 7],
                         [2, 2], [5, 6], [[1, 2, 5], [3, 7]]),
    # zero-draft seq alongside a normal one (bonus-only vs accepted).
    RejectionSamplerCase("zero_length_mixed", [1, 2], [1, 2], [0, 2], [5, 6],
                         [[5], [1, 2, 6]]),
    # whole batch has no draft tokens (bonus-only).
    RejectionSamplerCase("all_zero_length", [], [], [0, 0], [5, 6],
                         [[5], [6]]),
    # per-seq variable lengths, all accepted.
    RejectionSamplerCase("all_different_lengths", [1, 2, 3, 4, 5, 6],
                         [1, 2, 3, 4, 5, 6], [1, 2, 3], [7, 9, 10],
                         [[1, 7], [2, 3, 9], [4, 5, 6, 10]]),
    # large-K stress with a late mismatch.
    RejectionSamplerCase("single_long_sequence", list(range(1, 31)),
                         list(range(1, 28)) + [99, 29, 30], [30], [100],
                         [list(range(1, 28)) + [99]]),
]


def _target_logits_from_tokens(target_tokens: list[int], device):
    """Build [num_tokens, VOCAB_SIZE] logits whose per-row argmax is the
    desired target token id."""
    num_tokens = len(target_tokens)
    logits = torch.full((num_tokens, VOCAB_SIZE),
                        -100.0,
                        dtype=torch.float32,
                        device=device)
    for i, tok in enumerate(target_tokens):
        logits[i, tok] = 100.0
    return logits


def _segment_info(num_draft_per_seq: list[int], device):
    """Build segment_ids / group_indices the way the runner does, but with
    numpy. """
    counts = np.asarray(num_draft_per_seq, dtype=np.int64)
    seg = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    starts = np.zeros(len(counts), dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    grp = np.arange(seg.shape[0], dtype=np.int64) - np.repeat(starts, counts)
    return (torch.tensor(seg, dtype=torch.int64, device=device),
            torch.tensor(grp, dtype=torch.int64, device=device))


def _expected_matrix(expected_rows: list[list[int]], max_draft: int, device):
    """Pad the sibling's variable-length rows to the torch sampler's fixed
    [batch, max_draft + 1] output, filling with PLACEHOLDER_TOKEN_ID."""
    width = max_draft + 1
    padded = [
        row + [PLACEHOLDER_TOKEN_ID] * (width - len(row))
        for row in expected_rows
    ]
    return torch.tensor(padded, dtype=torch.int32, device=device)


def _synthetic_sampler(rates: list[float], device) -> RejectionSampler:
    speculative_config = SimpleNamespace(
        rejection_sample_method="synthetic",
        synthetic_acceptance_rates=rates,
    )
    return RejectionSampler(speculative_config, device)


@pytest.mark.parametrize("case", GREEDY_CASES, ids=lambda c: c.name)
def test_rejection_sampler_greedy_scenarios(case: RejectionSamplerCase,
                                            device):
    sampler = RejectionSampler()

    num_draft_tokens = torch.tensor(case.num_draft_per_seq,
                                    dtype=torch.int32,
                                    device=device)
    draft_token_ids = torch.tensor(case.draft_tokens,
                                   dtype=torch.int32,
                                   device=device)
    target_logits = _target_logits_from_tokens(case.target_tokens, device)
    bonus_token_ids = torch.tensor(case.bonus_tokens,
                                   dtype=torch.int32,
                                   device=device)

    segment_ids, group_indices = _segment_info(case.num_draft_per_seq, device)

    max_draft = max(case.num_draft_per_seq)
    output = sampler(draft_token_ids=draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=max_draft)

    expected = _expected_matrix(case.expected, max_draft, device)
    assert torch.equal(
        output,
        expected), (f"case '{case.name}': expected {expected.tolist()}, "
                    f"got {output.tolist()}")


def test_rejection_sampler_synthetic_converts_to_conditional_rates(device):
    sampler = _synthetic_sampler([0.8, 0.4, 0.0], device)

    assert sampler.synthetic_mode
    expected = torch.tensor([0.8, 0.5, 0.0], device=device)
    assert torch.allclose(sampler.synthetic_conditional_rates, expected)


def test_rejection_sampler_synthetic_greedy_uses_acceptance_schedule(device):
    sampler = _synthetic_sampler([1.0, 0.5, 0.0], device)
    draft_token_ids = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    target_logits = _target_logits_from_tokens([4, 5, 6], device)
    bonus_token_ids = torch.tensor([7], dtype=torch.int32, device=device)
    segment_ids, group_indices = _segment_info([3], device)
    accept_u = torch.tensor([0.99, 0.6, 0.0], device=device)

    output = sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=3,
        accept_u=accept_u,
    )

    expected = torch.tensor([[1, 5, -1, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_synthetic_rejects_placeholder_draft(device):
    sampler = _synthetic_sampler([1.0], device)
    draft_token_ids = torch.tensor([-1], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([1], dtype=torch.int32, device=device)
    target_logits = _target_logits_from_tokens([3], device)
    bonus_token_ids = torch.tensor([4], dtype=torch.int32, device=device)
    segment_ids, group_indices = _segment_info([1], device)

    output = sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=1,
        accept_u=torch.zeros(1, device=device),
    )

    expected = torch.tensor([[3, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


@pytest.mark.parametrize("padding_token_id", [-1, 0])
def test_rejection_sampler_greedy_ignores_padded_draft_slots(
        device, padding_token_id):
    """Ignore static-shape padding beyond the declared draft length."""
    sampler = RejectionSampler()
    draft_token_ids = torch.tensor([1, 2, padding_token_id, padding_token_id],
                                   dtype=torch.int32,
                                   device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    # Mismatched padding must not replace the real bonus token.
    target_logits = _target_logits_from_tokens([1, 2, 4, 5], device)
    bonus_token_ids = torch.tensor([3], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1, 2, 3],
                                 dtype=torch.int64,
                                 device=device)

    output = sampler(draft_token_ids=draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=3)

    expected = torch.tensor([[1, 2, 3, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_apply_top_k_top_p_top_k(device):
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]], device=device)
    top_k = torch.tensor([[2]], dtype=torch.int32, device=device)
    top_p = torch.tensor([[1.0]], dtype=torch.float32, device=device)

    out = apply_top_k_top_p(logits, top_k, top_p)

    assert torch.equal(out[0, 1:3], logits[0, 1:3])
    assert out[0, 0] == MASKED_LOGIT_VALUE
    assert out[0, 3] == MASKED_LOGIT_VALUE


def test_apply_top_k_top_p_top_k_with_negatives(device):
    # Regression guard for the bit-pattern binary search sign-bit handling:
    # with negative logits present, an incorrect total-order transform keeps the
    # wrong set. Top-3 of {1,3,2,0,-5,7} is {7,3,2} -> keep indices 1,2,5.
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0, -5.0, 7.0]], device=device)
    top_k = torch.tensor([[3]], dtype=torch.int32, device=device)
    top_p = torch.tensor([[1.0]], dtype=torch.float32, device=device)

    out = apply_top_k_top_p(logits, top_k, top_p)

    kept = out > MASKED_LOGIT_VALUE / 2
    expected_kept = torch.tensor([[False, True, True, False, False, True]],
                                 device=device)
    assert torch.equal(kept, expected_kept), out.tolist()
    assert torch.equal(out[0, [1, 2, 5]], logits[0, [1, 2, 5]])


def test_apply_top_k_top_p_top_p(device):
    probs = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32, device=device)
    logits = torch.log(probs)
    top_k = torch.tensor([[0]], dtype=torch.int32, device=device)
    top_p = torch.tensor([[0.7]], dtype=torch.float32, device=device)

    out = apply_top_k_top_p(logits, top_k, top_p)

    assert torch.equal(out[0, :2], logits[0, :2])
    assert out[0, 2] == MASKED_LOGIT_VALUE


def test_apply_top_k_top_p_neutral_is_noop(device):
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]], device=device)
    top_k = torch.tensor([[0]], dtype=torch.int32, device=device)
    top_p = torch.tensor([[1.0]], dtype=torch.float32, device=device)

    out = apply_top_k_top_p(logits, top_k, top_p)

    assert torch.equal(out, logits)


def test_rejection_sampler_random_all_accepted_uses_bonus(device):
    sampler = RejectionSampler()
    draft_token_ids = torch.tensor([1, 2], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    target_logits = torch.full((2, 5), -100.0, device=device)
    target_logits[0, 1] = 100.0
    target_logits[1, 2] = 100.0
    bonus_token_ids = torch.tensor([4], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
    temperatures = torch.ones((2, 1), dtype=torch.float32, device=device)
    top_k = torch.zeros((2, 1), dtype=torch.int32, device=device)
    top_p = torch.ones((2, 1), dtype=torch.float32, device=device)
    accept_u = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    recover_u = torch.full_like(target_logits, 0.5)

    output = sampler(draft_token_ids=draft_token_ids,
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
                     do_sampling=True)

    expected = torch.tensor([[1, 2, 4]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


@pytest.mark.parametrize("padding_token_id", [-1, 0])
def test_rejection_sampler_random_ignores_padded_draft_slots(
        device, padding_token_id):
    """Ignore static-shape padding beyond the declared draft length."""
    sampler = RejectionSampler()
    draft_token_ids = torch.tensor([1, 2, padding_token_id, padding_token_id],
                                   dtype=torch.int32,
                                   device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    target_logits = _target_logits_from_tokens([1, 2, 4, 5], device)
    bonus_token_ids = torch.tensor([3], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1, 2, 3],
                                 dtype=torch.int64,
                                 device=device)
    temperatures = torch.ones((4, 1), dtype=torch.float32, device=device)
    top_k = torch.zeros((4, 1), dtype=torch.int32, device=device)
    top_p = torch.ones((4, 1), dtype=torch.float32, device=device)

    output = sampler(draft_token_ids=draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=3,
                     temperatures=temperatures,
                     top_k=top_k,
                     top_p=top_p,
                     accept_u=torch.full((4, ), 0.01, device=device),
                     recover_u=torch.full_like(target_logits, 0.5),
                     do_sampling=True)

    expected = torch.tensor([[1, 2, 3, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_random_rejects_and_recovers(device):
    sampler = RejectionSampler()
    draft_token_ids = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    target_logits = torch.full((3, 6), -100.0, device=device)
    target_logits[0, 1] = 100.0
    target_logits[1, 2] = 0.0
    target_logits[1, 4] = 0.0
    target_logits[2, 3] = 100.0
    bonus_token_ids = torch.tensor([5], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    temperatures = torch.ones((3, 1), dtype=torch.float32, device=device)
    top_k = torch.zeros((3, 1), dtype=torch.int32, device=device)
    top_p = torch.ones((3, 1), dtype=torch.float32, device=device)
    accept_u = torch.tensor([0.01, 0.75, 0.01],
                            dtype=torch.float32,
                            device=device)
    recover_u = torch.full_like(target_logits, 0.5)

    output = sampler(draft_token_ids=draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=3,
                     temperatures=temperatures,
                     top_k=top_k,
                     top_p=top_p,
                     accept_u=accept_u,
                     recover_u=recover_u,
                     do_sampling=True)

    expected = torch.tensor([[1, 4, -1, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_synthetic_random_uses_acceptance_schedule(device):
    sampler = _synthetic_sampler([1.0, 0.0], device)
    draft_token_ids = torch.tensor([1, 2], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    target_logits = torch.full((2, 6), -100.0, device=device)
    target_logits[0, 3] = 100.0
    target_logits[1, 4] = 100.0
    bonus_token_ids = torch.tensor([5], dtype=torch.int32, device=device)
    segment_ids, group_indices = _segment_info([2], device)
    temperatures = torch.ones((2, 1), dtype=torch.float32, device=device)
    top_k = torch.ones((2, 1), dtype=torch.int32, device=device)
    top_p = torch.ones((2, 1), dtype=torch.float32, device=device)

    output = sampler(
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
        accept_u=torch.tensor([0.99, 0.0], device=device),
        recover_u=torch.full_like(target_logits, 0.5),
        do_sampling=True,
    )

    expected = torch.tensor([[1, 4, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_synthetic_random_placeholder_recovers_target(
        device):
    sampler = _synthetic_sampler([0.0], device)
    draft_token_ids = torch.tensor([-1], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([1], dtype=torch.int32, device=device)
    target_logits = torch.tensor([[-0.1053605, -2.3025851]],
                                 dtype=torch.float32,
                                 device=device)
    bonus_token_ids = torch.tensor([1], dtype=torch.int32, device=device)
    segment_ids, group_indices = _segment_info([1], device)

    output = sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=1,
        temperatures=torch.ones((1, 1), device=device),
        top_k=torch.zeros((1, 1), dtype=torch.int32, device=device),
        top_p=torch.ones((1, 1), device=device),
        accept_u=torch.zeros(1, device=device),
        recover_u=torch.full_like(target_logits, 0.5),
        do_sampling=True,
    )

    expected = torch.tensor([[0, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_synthetic_mixed_batch_recovers_greedy_argmax(
        device):
    sampler = _synthetic_sampler([0.0], device)
    draft_token_ids = torch.tensor([3, 1], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([1, 1], dtype=torch.int32, device=device)
    target_logits = torch.tensor(
        [[-100.0, -100.0, -100.0, 100.0, -100.0],
         [-100.0, -0.5108256, -100.0, -100.0, -0.9162907]],
        device=device,
    )
    bonus_token_ids = torch.tensor([4, 3], dtype=torch.int32, device=device)
    segment_ids, group_indices = _segment_info([1, 1], device)

    output = sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=1,
        temperatures=torch.tensor([[0.0], [1.0]], device=device),
        top_k=torch.zeros((2, 1), dtype=torch.int32, device=device),
        top_p=torch.ones((2, 1), device=device),
        accept_u=torch.zeros(2, device=device),
        recover_u=torch.full_like(target_logits, 0.5),
        do_sampling=True,
    )

    expected = torch.tensor([[3, -1], [4, -1]],
                            dtype=torch.int32,
                            device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_random_top_k_forces_recovery(device):
    sampler = RejectionSampler()
    draft_token_ids = torch.tensor([1], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([1], dtype=torch.int32, device=device)
    target_logits = torch.tensor([[0.0, 1.0, 2.0, 9.0]],
                                 dtype=torch.float32,
                                 device=device)
    bonus_token_ids = torch.tensor([2], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0], dtype=torch.int64, device=device)
    temperatures = torch.ones((1, 1), dtype=torch.float32, device=device)
    top_k = torch.tensor([[1]], dtype=torch.int32, device=device)
    top_p = torch.ones((1, 1), dtype=torch.float32, device=device)
    accept_u = torch.tensor([0.01], dtype=torch.float32, device=device)
    recover_u = torch.full_like(target_logits, 0.5)

    output = sampler(draft_token_ids=draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=1,
                     temperatures=temperatures,
                     top_k=top_k,
                     top_p=top_p,
                     accept_u=accept_u,
                     recover_u=recover_u,
                     do_sampling=True)

    expected = torch.tensor([[3, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_random_greedy_row_accepts_argmax(device):
    # A temperature-0 (greedy) request routed through the random path must
    # accept the draft token deterministically iff it equals the target argmax,
    # even when the raw softmax over the target logits is far from one-hot.
    # `accept_u` is pinned high (0.99) so a soft-softmax accept probability
    # (~0.52 here) would spuriously reject -- the temperature->one-hot collapse
    # is what makes this pass.
    sampler = RejectionSampler()
    draft_token_ids = torch.tensor([0, 3], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    target_logits = torch.tensor([[2.0, 1.9, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
                                 dtype=torch.float32,
                                 device=device)
    bonus_token_ids = torch.tensor([7], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
    temperatures = torch.zeros((2, 1), dtype=torch.float32, device=device)
    top_k = torch.zeros((2, 1), dtype=torch.int32, device=device)
    top_p = torch.ones((2, 1), dtype=torch.float32, device=device)
    accept_u = torch.tensor([0.99, 0.99], dtype=torch.float32, device=device)
    recover_u = torch.full_like(target_logits, 0.5)

    output = sampler(draft_token_ids=draft_token_ids,
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
                     do_sampling=True)

    expected = torch.tensor([[0, 3, 7]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_random_greedy_row_recovers_argmax(device):
    # A temperature-0 (greedy) request whose draft token is NOT the target
    # argmax must reject deterministically and recover the argmax, regardless of
    # `accept_u`. `accept_u` is pinned low (0.01) so a soft-softmax accept
    # probability (~0.48 for the drafted token) would spuriously accept.
    sampler = RejectionSampler()
    draft_token_ids = torch.tensor([1], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([1], dtype=torch.int32, device=device)
    target_logits = torch.tensor([[2.0, 1.9, 0.0, 0.0]],
                                 dtype=torch.float32,
                                 device=device)
    bonus_token_ids = torch.tensor([7], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0], dtype=torch.int64, device=device)
    temperatures = torch.zeros((1, 1), dtype=torch.float32, device=device)
    top_k = torch.zeros((1, 1), dtype=torch.int32, device=device)
    top_p = torch.ones((1, 1), dtype=torch.float32, device=device)
    accept_u = torch.tensor([0.01], dtype=torch.float32, device=device)
    recover_u = torch.full_like(target_logits, 0.5)

    output = sampler(draft_token_ids=draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=1,
                     temperatures=temperatures,
                     top_k=top_k,
                     top_p=top_p,
                     accept_u=accept_u,
                     recover_u=recover_u,
                     do_sampling=True)

    expected = torch.tensor([[0, -1]], dtype=torch.int32, device=device)
    assert torch.equal(output, expected)


def test_rejection_sampler_ignores_padded_segment_sentinel(device):
    sampler = RejectionSampler()
    draft_token_ids = torch.tensor([1, 0], dtype=torch.int32, device=device)
    num_draft_tokens = torch.tensor([1, 0], dtype=torch.int32, device=device)
    # The mismatched out-of-range padding segment must be ignored.
    target_logits = _target_logits_from_tokens([1, 7], device)
    bonus_token_ids = torch.tensor([9, 8], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 2], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 0], dtype=torch.int64, device=device)

    output = sampler(draft_token_ids=draft_token_ids,
                     num_draft_tokens=num_draft_tokens,
                     target_logits=target_logits,
                     bonus_token_ids=bonus_token_ids,
                     segment_ids=segment_ids,
                     group_indices=group_indices,
                     max_draft_tokens=1)

    expected = torch.tensor([[1, 9], [8, -1]],
                            dtype=torch.int32,
                            device=device)
    assert torch.equal(output, expected)
